"""COMP-CATOFF-2026-10 (Milan 2026-10-21): comparison star off its catalogue magnitude is excluded
and replaced from the reserve."""

from __future__ import annotations

import numpy as np

import pipeline  # noqa: F401  app import order (photometry modules)

from comp_smart import catalog_offsets, smart_select
from config import AppConfig

_RNG = np.random.default_rng(1)
N = 40
BJD = np.linspace(0.0, 0.1, N)
STAB = dict(max_comp_p2p=0.05, max_comp_slope_mmag_hr=50.0, comp_slope_significance_k=3.0,
            common_mode_detrend=True)


def _lc(cat: float, off: float = 0.0) -> np.ndarray:
    # instrumental = catalogue - zero point (+ offset), common transparency, small noise
    return cat - 20.0 + off + 0.02 * np.sin(BJD * 30) + _RNG.normal(0, 0.003, N)


def test_offsets_find_the_wrong_star() -> None:
    cat = {f"c{i}": 10.0 + 0.3 * i for i in range(6)}
    lc = {c: _lc(m) for c, m in cat.items()}
    lc["c2"] = _lc(cat["c2"], off=-0.6)  # measured 0.6 mag brighter than the catalogue says
    off = catalog_offsets(lc, cat)
    assert abs(off["c2"] - 0.6) < 0.02
    assert all(abs(v) < 0.02 for c, v in off.items() if c != "c2")


def test_too_few_band_mags_no_test() -> None:
    cat = {"a": 10.0, "b": 11.0, "c": float("nan")}
    assert catalog_offsets({c: _lc(10.0) for c in cat}, cat) == {}


def test_smart_select_replaces_excluded_from_reserve() -> None:
    prim = [f"p{i}" for i in range(5)]
    res = ["r0", "r1"]
    cat = {c: 10.0 + 0.2 * i for i, c in enumerate(prim + res)}
    lc = {c: _lc(cat[c]) for c in cat}
    lc["p1"] = _lc(cat["p1"], off=0.6)
    bjd = {c: BJD for c in lc}
    final, log = smart_select(prim, res, lc, bjd, n_comp_min=3, n_comp_max=5, stability_kwargs=STAB,
                              cat_mag=cat, catalog_offset_max_mag=0.20)
    assert "p1" not in final and len(final) == 5 and "r0" in final
    assert log["verdict"]["p1"] == "excluded" and "catalogue offset" in log["notes"]["p1"]
    assert log["catalog_offset"]["excluded"] == ["p1"]
    # off -> unchanged selection
    final0, log0 = smart_select(prim, res, lc, bjd, n_comp_min=3, n_comp_max=5, stability_kwargs=STAB,
                                cat_mag=cat, catalog_offset_max_mag=0.0)
    assert "catalog_offset" not in log0


def test_config_default_and_range() -> None:
    cfg = AppConfig()
    assert cfg.comp_catalog_offset_max_mag == 0.20


def test_catalog_offset_excluded_star_is_no_check_star() -> None:
    import pandas as pd

    from check_star_kmag import drop_smart_rejected_check_candidates

    pool = pd.DataFrame({"catalog_id": ["111", "222", "333"]})
    out = drop_smart_rejected_check_candidates(pool, {"replaced": ["111"]}, "t",
                                               catalog_offset_excluded={"333"})
    assert out["catalog_id"].tolist() == ["222"]
