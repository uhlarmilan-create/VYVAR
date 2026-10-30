"""NIGHT-FWHM-2026-10: one FWHM for neighbour geometry - the group's night median."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd

import photometry_core  # noqa: F401  (import order: avoids circular import)
from photometry_comp import night_median_fwhm_px

SRC = Path(__file__).resolve().parents[2] / "src_py"


def _write_group(d: Path, fwhms: list[float]) -> list[Path]:
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for i, f in enumerate(fwhms):
        p = d / f"proc_{i:03d}.csv"
        pd.DataFrame({"x": [1.0, 2.0], "fwhm_px_for_aperture": [f, f]}).to_csv(p, index=False)
        out.append(p)
    return out


def test_group_median_ignores_other_filter(tmp_path) -> None:
    g = _write_group(tmp_path / "g", [5.2, 5.31, 5.4, 6.0, 5.0])
    r = _write_group(tmp_path / "r", [4.8, 4.91, 5.0])
    fg, sg = night_median_fwhm_px(g, 4.64)
    fr, _ = night_median_fwhm_px(r, 4.64)
    assert math.isclose(fg, 5.31) and math.isclose(fr, 4.91)
    assert "night median of 5 frames" in sg
    mixed, _ = night_median_fwhm_px(g + r, 4.64)
    assert not math.isclose(mixed, fg)  # the draft-level mix is a different number


def test_fallback_nan_and_missing_column(tmp_path) -> None:
    p = tmp_path / "proc_000.csv"
    pd.DataFrame({"x": [1.0]}).to_csv(p, index=False)
    v, src = night_median_fwhm_px([p], float("nan"))
    assert math.isnan(v) and "fallback" in src
    v, src = night_median_fwhm_px([], 4.64)
    assert v == 4.64 and "fallback" in src


def test_isolation_rule_uses_night_fwhm_units() -> None:
    # _nn_dist_fwhm and the threshold must use the same FWHM (comp_selection_per_target)
    s = (SRC / "comp_selection_per_target.py").read_text(encoding="utf-8")
    assert 'candidate_pool_df["_nn_dist_fwhm"] = nn_px / _fw_nn' in s
    sel = _calls(s, "_select_comps_by_rms_then_color")
    kw = {k.arg: k.value for k in sel[0].keywords}
    assert "_fw_nn" in ast.unparse(kw["fwhm_px"])


def _calls(src: str, func: str) -> list[ast.Call]:
    tree = ast.parse(src)
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", None)) == func]


def test_comp_selection_wiring_uses_night_fwhm() -> None:
    s = (SRC / "target_comp_selection.py").read_text(encoding="utf-8")
    sel = _calls(s, "select_comparison_stars_per_target")
    assert len(sel) == 1
    kw = {k.arg: k.value for k in sel[0].keywords}
    assert getattr(kw["fwhm_px"], "id", None) == "fwhm_night_px"
    assert "_k_md * float(fwhm_night_px)" in s
    # D3 astrometric gate stays on the MASTERSTAR FWHM (measured on that image)
    pool = _calls(s, "build_global_comp_pool")
    kwp = {k.arg: k.value for k in pool[0].keywords}
    assert ast.unparse(kwp["fwhm_px"]) == "float(fwhm_px)"
    assert len(_calls(s, "night_median_fwhm_px")) == 1


def test_aperture_uses_group_night_fwhm() -> None:
    s = (SRC / "aperture_photometry_state.py").read_text(encoding="utf-8")
    c = _calls(s, "night_median_fwhm_px")
    assert len(c) == 1 and ast.unparse(c[0].args[0]) == "csv_files"
    i_qc = s.index("load_qc_fwhm_map(_qc_csv_p2a)")
    i_grp = s.index("night_median_fwhm_px(", i_qc)
    i_use = s.index("fwhm_night_median_px=_night_p2a", i_grp)
    assert i_qc < i_grp < i_use


def test_median_robust_to_bad_seeing_tail(tmp_path) -> None:
    vals = list(np.full(20, 5.3)) + [9.0, 10.0, 11.0]
    g = _write_group(tmp_path / "g", vals)
    v, _ = night_median_fwhm_px(g, 4.64)
    assert math.isclose(v, 5.3)


def test_d3_gate_runs_once_on_the_pool() -> None:
    """G07-25: D3 runs once, on the global pool (MASTERSTAR FWHM + plate-solve residual), not per target."""
    import inspect

    from comp_selection_per_target import _build_candidates_pre

    assert "apply_d3_comparison_candidacy" not in inspect.getsource(_build_candidates_pre)
    pc = (SRC / "photometry_comp.py").read_text(encoding="utf-8")
    assert pc.count("apply_d3_comparison_candidacy(") == 1
