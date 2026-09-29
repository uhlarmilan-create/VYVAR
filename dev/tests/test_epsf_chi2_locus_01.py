# -*- coding: ascii -*-
"""EPSF-CHI2-LOCUS-01: locus SET for psf_fit_ok (T1-T4).

Pre-fail / post-pass: at 5a5b07a the fixed threshold=50 rejected bright
on-locus stars; after the locus wire they pass. Outliers and nonfinite
chi2 stay rejected. Too-few-star frames use the night locus path.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from psf_chi2_locus import (
    MIN_N_FRAME_LOCUS,
    Chi2Locus,
    apply_chi2_locus_to_rows,
    chi2_ok_from_locus,
    fit_chi2_locus,
    prefer_night_locus,
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
) -> list[dict]:
    rng = np.random.default_rng(seed)
    # Flux span covering the former chi2=50 crossing (~2e5) and brighter.
    logf = rng.uniform(3.5, 6.5, size=n)
    flux = 10**logf
    logc = a + k * logf + rng.normal(0.0, scatter, size=n)
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
    # Inject high-flux on-locus stars that would fail fixed threshold=50.
    for j, lf in enumerate((5.5, 5.8, 6.0, 6.2)):
        f = 10**lf
        c = 10 ** (a + k * lf)  # exactly on locus, chi2 >> 50 when bright
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
        # 10-sigma above locus in log10 space
        c = float(10 ** (a + k * math.log10(f) + outlier_sigma * scatter * 1.4826))
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
    # Pre-fix: bright on-locus stars with chi2>=50 fail.
    bright = [r for r in rows if str(r["catalog_id"]).startswith("BRIGHT")]
    assert bright, "expected injected bright stars"
    for r in bright:
        assert r["psf_chi2"] >= 50.0
        assert _legacy_fit_ok(r["psf_chi2"], True) is False

    # Fit night locus from the synthetic field (same generative model).
    night = fit_chi2_locus(
        [r["psf_flux"] for r in rows],
        [r["psf_chi2"] for r in rows],
        converged=[True] * len(rows),
        source="night",
    )
    assert night is not None
    out, locus, meta = apply_chi2_locus_to_rows(rows, night_locus=night, n_sigma=5.0)
    assert locus is not None
    assert meta["source"] == "night"  # frame n << MIN_N_FRAME_LOCUS
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
        # Pre-fix absolute cut may or may not catch faint outliers; locus must.
        assert by_id[str(r["catalog_id"])]["psf_fit_ok"] is False
        assert by_id[str(r["catalog_id"])]["psf_chi2_locus_resid_sigma"] > 5.0
        # And they remain False under the legacy absolute cut when chi2>=50.
        if r["psf_chi2"] >= 50.0:
            assert _legacy_fit_ok(r["psf_chi2"], True) is False


def test_t3_nonfinite_chi2_fails_both_paths():
    # Shared SET function used by both iterative and grouped paths.
    assert chi2_ok_from_locus(converged=True, chi2=float("nan"), resid_sigma=0.0) is False
    assert chi2_ok_from_locus(converged=True, chi2=float("inf"), resid_sigma=0.0) is False
    # Pre-fix grouped path would have PASSed nonfinite chi2; document the flip.
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


def test_t4_too_few_stars_uses_night_locus_never_fixed_threshold():
    # Tiny frame: prefer_night must be True.
    tiny = _synthetic_frame(n=12, seed=3)
    assert len(tiny) < MIN_N_FRAME_LOCUS
    frame_only = fit_chi2_locus(
        [r["psf_flux"] for r in tiny],
        [r["psf_chi2"] for r in tiny],
        converged=[True] * len(tiny),
        source="frame",
    )
    assert prefer_night_locus(frame_only) is True

    night = Chi2Locus(a=-3.9, k=1.06, scatter=0.05, n=5000, source="night", k_err=0.001)
    out, locus, meta = apply_chi2_locus_to_rows(tiny, night_locus=night, n_sigma=5.0)
    assert meta["source"] == "night"
    assert locus is not None
    assert locus.source == "night" or meta["source"] == "night"
    # Without night locus: must fail closed (no fixed threshold fallback).
    out2, locus2, meta2 = apply_chi2_locus_to_rows(tiny, night_locus=None, n_sigma=5.0)
    assert meta2["source"] == "none_need_night"
    assert locus2 is None
    assert all(r["psf_fit_ok"] is False for r in out2)


def test_min_n_frame_locus_derived_from_sat_chi2():
    # n where k_err = 3 * night k_err => (n_night-1)/9 + 1
    assert MIN_N_FRAME_LOCUS == 1 + math.ceil((30076 - 1) / 9.0)


def test_finalize_night_locus_refreshes_n_ok_for_invariant(tmp_path):
    """Night finalize must run before INV-PSF-FRAME-01 (merge leaves n_ok=0)."""
    from psf_chi2_locus import finalize_night_locus_for_inv_psf_frame_01

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
    import pandas as pd

    pd.DataFrame(rows).to_csv(proc, index=False)
    recs = [{"frame_name": "BO_CVn_Light_001.fits", "n_fit": 20, "n_ok": 0}]
    summary = finalize_night_locus_for_inv_psf_frame_01(tmp_path, recs, n_sigma=5.0)
    assert summary.get("n_rewritten", 0) >= 1 or summary.get("n_kept_frame", 0) >= 1
    assert recs[0]["n_ok"] == 20
    assert summary.get("n_ok_refreshed_records") == 1
