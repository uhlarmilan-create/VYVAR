# -*- coding: ascii -*-
"""EPSF-CHI2-LOCUS-01: locus SET for psf_fit_ok (T1-T4 + Phase 1b).

Pre-fail / post-pass: at 5a5b07a the fixed threshold=50 rejected bright
on-locus stars; after the locus wire they pass. Outliers and nonfinite
chi2 stay rejected. Phase 1b: night k + per-frame a_f; no night-specific
constants; structural drop of never-fitted comps.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from psf_chi2_locus import (
    MIN_N_LOCUS_FIT,
    Chi2Locus,
    apply_chi2_locus_to_rows,
    chi2_ok_from_locus,
    fit_chi2_locus,
    frame_intercept_with_night_k,
)


def _synthetic_frame(
    *,
    n: int = 80,
    k: float = 1.06,
    a: float = -3.9,
    scatter: float = 0.05,
    seed: int = 0,
    n_outliers: int = 0,
    outlier_sigma: float = 10.0,
    a_offset: float = 0.0,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    logf = rng.uniform(3.5, 6.5, size=n)
    flux = 10**logf
    logc = a + a_offset + k * logf + rng.normal(0.0, scatter, size=n)
    chi2 = 10**logc
    rows = []
    for i in range(n):
        rows.append(
            {
                "catalog_id": f"S{i}",
                "psf_flux": float(flux[i]),
                "psf_chi2": float(chi2[i]),
                "psf_converged": True,
                "psf_fit_ok": False,
            }
        )
    for j, lf in enumerate((5.5, 5.8, 6.0, 6.2)):
        f = 10**lf
        c = 10 ** (a + a_offset + k * lf)
        rows.append(
            {
                "catalog_id": f"BRIGHT{j}",
                "psf_flux": float(f),
                "psf_chi2": float(c),
                "psf_converged": True,
                "psf_fit_ok": False,
            }
        )
    for j in range(n_outliers):
        f = float(10 ** rng.uniform(4.0, 5.5))
        c = float(10 ** (a + a_offset + k * math.log10(f) + outlier_sigma * scatter * 1.4826))
        rows.append(
            {
                "catalog_id": f"OUT{j}",
                "psf_flux": f,
                "psf_chi2": c,
                "psf_converged": True,
                "psf_fit_ok": False,
            }
        )
    return rows


def _legacy_fit_ok(chi2: float, converged: bool, threshold: float = 50.0) -> bool:
    """Pre-fix SET (iterative path at 5a5b07a)."""
    return bool(converged and math.isfinite(chi2) and chi2 < threshold)


def test_t1_bright_on_locus_passes_after_locus_fails_fixed_50():
    rows = _synthetic_frame(n=120, seed=1)
    bright = [r for r in rows if str(r["catalog_id"]).startswith("BRIGHT")]
    assert bright, "expected injected bright stars"
    for r in bright:
        assert r["psf_chi2"] >= 50.0
        assert _legacy_fit_ok(r["psf_chi2"], True) is False

    night = fit_chi2_locus(
        [r["psf_flux"] for r in rows],
        [r["psf_chi2"] for r in rows],
        converged=[True] * len(rows),
        source="night",
    )
    assert night is not None
    out, locus, meta = apply_chi2_locus_to_rows(rows, night_locus=night, n_sigma=5.0)
    assert locus is not None
    assert meta["source"] == "frame"  # n_f >= MIN_N_LOCUS_FIT -> per-frame a_f
    by_id = {str(r["catalog_id"]): r for r in out}
    for r in bright:
        got = by_id[str(r["catalog_id"])]
        assert got["psf_fit_ok"] is True, (
            f"{r['catalog_id']} chi2={got['psf_chi2']:.1f} "
            f"resid={got['psf_chi2_locus_resid_sigma']:.2f} source={meta['source']}"
        )


def test_t2_injected_outliers_fail_before_and_after():
    rows = _synthetic_frame(n=200, seed=2, n_outliers=5, outlier_sigma=10.0)
    outs = [r for r in rows if str(r["catalog_id"]).startswith("OUT")]
    assert len(outs) == 5

    night = fit_chi2_locus(
        [r["psf_flux"] for r in rows if not str(r["catalog_id"]).startswith("OUT")],
        [r["psf_chi2"] for r in rows if not str(r["catalog_id"]).startswith("OUT")],
        converged=[True] * (len(rows) - 5),
        source="night",
    )
    assert night is not None
    out, _, _ = apply_chi2_locus_to_rows(rows, night_locus=night, n_sigma=5.0)
    by_id = {str(r["catalog_id"]): r for r in out}
    for r in outs:
        assert by_id[str(r["catalog_id"])]["psf_fit_ok"] is False
        assert by_id[str(r["catalog_id"])]["psf_chi2_locus_resid_sigma"] > 5.0
        if r["psf_chi2"] >= 50.0:
            assert _legacy_fit_ok(r["psf_chi2"], True) is False


def test_t3_nonfinite_chi2_fails_both_paths():
    assert chi2_ok_from_locus(converged=True, chi2=float("nan"), resid_sigma=0.0) is False
    assert chi2_ok_from_locus(converged=True, chi2=float("inf"), resid_sigma=0.0) is False
    legacy_grouped_ok = (not math.isfinite(float("nan"))) or (float("nan") < 50.0)
    assert legacy_grouped_ok is True
    rows = [
        {
            "catalog_id": "NF",
            "psf_flux": 1.0e5,
            "psf_chi2": float("nan"),
            "psf_converged": True,
            "psf_fit_ok": False,
        },
        {
            "catalog_id": "OK",
            "psf_flux": 1.0e4,
            "psf_chi2": 5.0,
            "psf_converged": True,
            "psf_fit_ok": False,
        },
    ]
    night = Chi2Locus(a=-3.9, k=1.06, scatter=0.1, n=100, source="night")
    out, _, _ = apply_chi2_locus_to_rows(rows, night_locus=night, n_sigma=5.0)
    by_id = {str(r["catalog_id"]): r for r in out}
    assert by_id["NF"]["psf_fit_ok"] is False
    assert by_id["OK"]["psf_fit_ok"] is True


def test_t4a_frame_global_chi2_offset_stays_on_locus_via_a_f():
    """Seeing-like global chi2 offset: a_f absorbs it; stars stay fit_ok."""
    night_rows = _synthetic_frame(n=200, seed=10, a=-3.9)
    night = fit_chi2_locus(
        [r["psf_flux"] for r in night_rows],
        [r["psf_chi2"] for r in night_rows],
        converged=[True] * len(night_rows),
        source="night",
    )
    assert night is not None
    # Frame with same k but intercept shifted by ~2 night scatters.
    offset = 2.0 * night.scatter
    frame = _synthetic_frame(n=80, seed=11, a=-3.9, a_offset=offset)
    out, locus, meta = apply_chi2_locus_to_rows(frame, night_locus=night, n_sigma=5.0)
    assert meta["source"] == "frame"
    assert locus is not None
    assert abs(locus.a - (night.a + offset)) < 0.15
    assert abs(locus.k - night.k) < 1e-12
    n_ok = sum(1 for r in out if r["psf_fit_ok"])
    assert n_ok >= int(0.9 * len(frame))


def test_t4b_below_min_n_uses_night_intercept():
    tiny = _synthetic_frame(n=5, seed=3)
    assert len(tiny) < MIN_N_LOCUS_FIT
    night = Chi2Locus(a=-3.9, k=1.06, scatter=0.05, n=5000, source="night")
    out, locus, meta = apply_chi2_locus_to_rows(tiny, night_locus=night, n_sigma=5.0)
    assert meta["source"] == "night"
    assert locus is not None
    assert locus.a == night.a
    assert locus.source == "night"
    out2, locus2, meta2 = apply_chi2_locus_to_rows(tiny, night_locus=None, n_sigma=5.0)
    assert meta2["source"] == "none_need_night"
    assert locus2 is None
    assert all(r["psf_fit_ok"] is False for r in out2)


def test_t4c_no_night_specific_numeric_constants():
    """psf_chi2_locus.py must not embed one-night / one-rig measure constants."""
    src = Path(__file__).resolve().parents[1].parent / "src_py" / "psf_chi2_locus.py"
    text = src.read_text(encoding="ascii")
    banned = (
        "0.002668",
        "0.336561",
        "30076",
        "SAT_CHI2",
        "MIN_N_FRAME_LOCUS",
        "3343",
    )
    for tok in banned:
        assert tok not in text, f"night-specific constant leaked: {tok}"
    # No float literals that look like the SAT-CHI2 night k_err / scatter.
    for m in re.finditer(r"(?<![A-Za-z_])\d+\.\d{4,}", text):
        val = float(m.group(0))
        assert not (0.002 < val < 0.003), f"suspicious k_err-like literal {val}"
        assert not (0.33 < val < 0.34), f"suspicious scatter-like literal {val}"


def test_finalize_night_locus_refreshes_n_ok_for_invariant(tmp_path):
    """Night finalize must run before INV-PSF-FRAME-01 (merge leaves n_ok=0)."""
    from psf_chi2_locus import finalize_night_locus_for_inv_psf_frame_01
    import pandas as pd

    proc = tmp_path / "proc_BO_CVn_Light_001.csv"
    rows = []
    for i in range(20):
        f = 10 ** (4.0 + i * 0.05)
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
    recs = [{"frame_name": "BO_CVn_Light_001.fits", "n_fit": 20, "n_ok": 0}]
    summary = finalize_night_locus_for_inv_psf_frame_01(tmp_path, recs, n_sigma=5.0)
    assert summary.get("n_kept_frame", 0) >= 1 or summary.get("n_rewritten", 0) >= 1
    assert recs[0]["n_ok"] == 20
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
            "catalog_id": ["A", "A", "B", "B", "C", "C"],
            "psf_flux": [100.0, 110.0, 200.0, 210.0, float("nan"), float("nan")],
            "psf_chi2": [1.0, 1.1, 2.0, 2.1, float("nan"), float("nan")],
            "psf_fit_ok": [True, True, True, True, False, False],
        }
    )
    measured = comps_with_psf_measurement_on_night(stack)
    assert measured == {"A", "B"}
    assert "C" not in measured
    # Pre-fix style: INV-PIN would drop every epoch because C never has flux.
    pins = ["A", "B", "C"]
    weights = {"A": 0.4, "B": 0.4, "C": 0.2}
    kept, w2, dropped = filter_ensemble_to_psf_measured(pins, weights, measured)
    assert kept == ["A", "B"]
    assert dropped == ["C"]
    assert abs(sum(w2.values()) - 1.0) < 1e-9
    assert "C" not in w2
