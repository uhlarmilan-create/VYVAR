"""COMP-POOL-ONLY-2026-10 (D5): comparison stars only from the rule pool; a red target with comps beyond
the colour cap keeps its light curve but is trust RED."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import pipeline  # noqa: F401  app import order (import cycles, T13)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import AppConfig
from photometry_comp import select_comparison_stars_per_target
from trust_flag_core import CompTrustThresholds, classify_warnings


def test_empty_pool_gives_no_comps_never_masterstars():
    from test_comp_determinism_synthetic import _TARGET_ID, _build_synthetic_dataset

    target, ms, paths, cache = _build_synthetic_dataset()
    cfg = AppConfig()
    cfg.gs11_dilution_enabled = False
    out = select_comparison_stars_per_target(
        target, ms, paths, csv_cache=cache, cfg=cfg, chip_fw=800, chip_fh=600, chip_interior_margin_px=0,
        min_dist_arcsec=30.0, variable_target_catalog_ids=frozenset({_TARGET_ID}),
        global_comp_pool_df=pd.DataFrame(),
        fwhm_px=3.7, n_comp_min=3, n_comp_max=7, max_comp_rms=0.1, min_frames_frac=0.3, plate_scale_arcsec=1.3,
    )
    assert out.empty


def test_comps_beyond_the_colour_cap_are_red():
    kw = dict(n_clean=5, check_scatter=0.01, lc_quality="good", thresholds=CompTrustThresholds.from_bounds(3, 8),
              n_check=100, check_min_epochs=5)
    hard, _ = classify_warnings(comp_delta_bprp_max=1.4, comp_delta_bprp_cap=0.79, **kw)
    assert any("colour mismatch" in h for h in hard)
    hard, _ = classify_warnings(comp_delta_bprp_max=0.5, comp_delta_bprp_cap=0.79, **kw)
    assert not any("colour mismatch" in h for h in hard)
