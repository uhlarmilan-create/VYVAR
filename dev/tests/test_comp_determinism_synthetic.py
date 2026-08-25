"""
Synthetic regression test for comparison-star selection determinism (P0-P4).

Exercises ``photometry_core.select_comparison_stars_per_target`` without archive
draft data. Uses short ``catalog_id`` strings (G001...) so Gaia ID normalization in
the pipeline matches per-frame CSV rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import comp_selection_per_target  # noqa: F401 - determinism modules under test
import comp_pool_rms  # noqa: F401
from config import AppConfig
from photometry_core import select_comparison_stars_per_target

_N_STARS = 30
_N_FRAMES = 20
_TARGET_ID = "G001"
_RNG_SEED = 42


def _build_synthetic_dataset(
    *,
    shuffle_masterstars: bool = False,
) -> tuple[pd.Series, pd.DataFrame, list[Path], dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(_RNG_SEED)
    ids = [f"G{i:03d}" for i in range(1, _N_STARS + 1)]

    rows: list[dict[str, object]] = []
    for i, sid in enumerate(ids):
        ang = (i + 1) * 0.04
        mag = 12.0 + (i % 6) * 0.15
        flux = 60_000.0 + i * 800.0
        rows.append(
            {
                "catalog_id": sid,
                "name": sid,
                "source_id": sid,
                "ra_deg": 180.0 + ang,
                "dec_deg": 45.0 + ang * 0.3,
                "x": 150.0 + (i % 10) * 55.0,
                "y": 150.0 + (i // 10) * 45.0,
                "phot_g_mean_mag": mag,
                "mag": mag,
                "bp_rp": 1.02,
                "dao_flux": flux,
                "flux": flux,
                "is_usable": True,
                "is_saturated": False,
                "is_noisy": False,
                "snr50_ok": True,
                "vsx_known_variable": False,
                "likely_saturated": False,
                "zone": "linear",
                "source_state": "DETECTED_P1",
                "vy_identity_gate": "ok",
                "gaia_dao_resid_px": 0.2,
                "snr": 80.0,
            }
        )

    masterstars = pd.DataFrame(rows)
    if shuffle_masterstars:
        masterstars = masterstars.sample(frac=1.0, random_state=_RNG_SEED + 1).reset_index(
            drop=True
        )

    target = masterstars.loc[masterstars["catalog_id"] == _TARGET_ID].iloc[0]

    flux_by_id = masterstars.set_index("catalog_id", drop=False)
    per_frame_paths: list[Path] = []
    csv_cache: dict[str, pd.DataFrame] = {}
    for fi in range(_N_FRAMES):
        path = Path(f"synthetic_proc_{fi:03d}.csv")
        per_frame_paths.append(path)
        frame_rows: list[dict[str, object]] = []
        # Stable catalog_id order so frame CSVs do not depend on masterstars row order.
        for sid in sorted(ids):
            star = flux_by_id.loc[sid]
            base = float(star["dao_flux"])
            flux = base * (1.0 + 1e-5 * float(rng.standard_normal()))
            frame_rows.append(
                {
                    "name": sid,
                    "catalog_id": sid,
                    "bjd_tdb_mid": 2_459_000.0 + fi * 0.01,
                    "dao_flux": flux,
                    "flux": flux,
                    "mag": float(star["mag"]),
                    "noise_floor_adu": 40.0,
                    "aperture_r_px": 7.0,
                    "is_usable": True,
                    "is_saturated": False,
                    "is_noisy": False,
                    "snr50_ok": True,
                    "vsx_known_variable": False,
                    "likely_saturated": False,
                }
            )
        csv_cache[str(path)] = pd.DataFrame(frame_rows)

    return target, masterstars, per_frame_paths, csv_cache


def _run_comp_selection(
    target: pd.Series,
    masterstars: pd.DataFrame,
    per_frame_paths: list[Path],
    csv_cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    cfg = AppConfig()
    cfg.gs11_dilution_enabled = False
    out = select_comparison_stars_per_target(
        target,
        masterstars,
        per_frame_paths,
        csv_cache=csv_cache,
        cfg=cfg,
        chip_fw=800,
        chip_fh=600,
        chip_interior_margin_px=0,
        max_psf_chi2=float("inf"),
        max_fwhm_factor=float("inf"),
        min_dist_arcsec=30.0,
        variable_target_catalog_ids=frozenset({_TARGET_ID}),
    )
    assert not out.empty, "synthetic fixture must select at least one comp star"
    return out.reset_index(drop=True)


def test_comp_selection_identical_on_repeated_calls():
    """Two runs with identical inputs must yield the same output frame."""
    target, masterstars, paths, cache = _build_synthetic_dataset()
    out_a = _run_comp_selection(target, masterstars, paths, cache)
    out_b = _run_comp_selection(target, masterstars, paths, cache)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_comp_selection_invariant_to_masterstars_row_order():
    """Shuffled masterstars row order must not change the selected comp set."""
    target, masterstars, paths, cache = _build_synthetic_dataset()
    out_ordered = _run_comp_selection(target, masterstars, paths, cache)

    _, masterstars_shuf, paths2, cache2 = _build_synthetic_dataset(shuffle_masterstars=True)
    out_shuffled = _run_comp_selection(target, masterstars_shuf, paths2, cache2)

    assert set(out_ordered["catalog_id"]) == set(out_shuffled["catalog_id"])
