"""EPSF-TARGETS-2026-10: ePSF tab helpers (target list of the PSF light curves, light-curve figure)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ui_epsf_dashboard import epsf_run_plan, psf_lc_figure, psf_lightcurve_targets


def test_targets_with_psf_lc_carry_name_type_and_mag(tmp_path: Path) -> None:
    phot = tmp_path / "photometry"
    lc = phot / "lightcurves"
    lc.mkdir(parents=True)
    pd.DataFrame({"catalog_id": ["1", "2"], "vsx_name": ["V1 Cyg", "V2 Cyg"], "vsx_type": ["EW", "RS"],
                  "mag": [14.2, 12.1], "catalog": ["VSX", "VSX"], "skip_photometry": [False, False]}).to_csv(
        phot / "active_targets.csv", index=False)
    for cid in ("1", "2"):
        (lc / f"lightcurve_{cid}.csv").write_text("bjd\n")
    (lc / "lightcurve_1_psf.csv").write_text("bjd\n")
    t = psf_lightcurve_targets(tmp_path)
    assert t.to_dict("records") == [{"catalog_id": "1", "name": "V1 Cyg", "vsx_type": "EW", "mag": 14.2}]


def test_figure_shifts_psf_onto_the_aperture_level_only_with_the_aperture_curve() -> None:
    n = 20
    df = pd.DataFrame({"bjd": 2461316.4 + np.arange(n) * 0.01, "ap_mag": np.full(n, 0.5), "ap_err": 0.01,
                       "ap_ok": True, "psf_mag": np.full(n, 0.3), "psf_err": 0.008, "psf_ok": True})
    with_ap = psf_lc_figure(df, "T", show_aperture=True)
    assert [t.name for t in with_ap.data] == ["aperture", "ePSF"]
    assert np.allclose(with_ap.data[1].y, 0.5)  # PSF level moved onto the aperture level
    alone = psf_lc_figure(df, "T", show_aperture=False)
    assert [t.name for t in alone.data] == ["ePSF"] and np.allclose(alone.data[0].y, 0.3)
    assert with_ap.layout.yaxis.autorange == "reversed"


def _setup(root: Path, name: str, types: list[str], *, frames: bool = True) -> dict:
    og = root / name
    phot = og / "photometry"
    (phot / "lightcurves").mkdir(parents=True)
    ids = [str(i + 1) for i in range(len(types))]
    pd.DataFrame({"catalog_id": ids, "vsx_name": [f"V{i} Cyg" for i in ids], "vsx_type": types, "mag": 13.0,
                  "catalog": "VSX", "skip_photometry": False}).to_csv(phot / "active_targets.csv", index=False)
    for cid in ids:
        (phot / "lightcurves" / f"lightcurve_{cid}.csv").write_text("bjd\n")
    pd.DataFrame({"target_catalog_id": ids, "catalog_id": [f"c{i}" for i in ids]}).to_csv(
        phot / "comparison_stars_per_target.csv", index=False)
    (og / "MASTERSTAR.fits").write_text("x")
    (og / "masterstars_full_match.csv").write_text("x")
    fr = root / f"frames_{name}"
    if frames:
        fr.mkdir()
    return {"obs_group_dir": og, "masterstar_fits": og / "MASTERSTAR.fits", "per_frame_csv_dir": fr}


def test_run_plan_covers_every_setup_with_every_in_scope_target(tmp_path: Path) -> None:
    """Every target with an aperture light curve of each setup; vsx_out_of_scope_types excludes (S6-FIX)."""
    setups = {"g_120_4": _setup(tmp_path, "g", ["EA", "RRAB"]), "r_100_4": _setup(tmp_path, "r", ["EA", "EW"]),
              "i_60_4": _setup(tmp_path, "i", ["EA"], frames=False)}
    plan = {r.setup: r for r in epsf_run_plan(setups, ["RRAB"])}
    assert set(plan) == set(setups)
    assert sorted(plan["g_120_4"].fit.target_ids) == ["1"] and plan["g_120_4"].runnable
    assert sorted(plan["r_100_4"].fit.target_ids) == ["1", "2"] and plan["r_100_4"].runnable
    assert plan["i_60_4"].missing == ("aligned frames directory",) and not plan["i_60_4"].runnable
    assert plan["r_100_4"].platesolve_dir == tmp_path / "r"


def test_generate_files_writes_every_setup_with_epsf_curves(tmp_path: Path) -> None:
    """EPSF-EXPORT-2026-10: the tab's button writes the user files of each setup from its ePSF light curves."""
    from ui_epsf_dashboard import generate_epsf_files

    setups = {"g_120_4": _setup(tmp_path, "g", ["EA", "EW"]), "r_100_4": _setup(tmp_path, "r", ["EA"])}
    lc = tmp_path / "g" / "photometry" / "lightcurves"
    (lc / "lightcurve_1_psf.csv").write_text("# ensemble_n_comp=3\n# psf_lc_time_base=BJD_TDB\nbjd,psf_fit_ok,psf_delta_mag,psf_delta_mag_err\n"
                                             "2461316.3,True,0.5,0.01\n2461316.4,True,0.6,0.01\n")
    assert generate_epsf_files(setups, "UMIA") == {"g_120_4": 1}
    out = list((tmp_path / "g" / "photometry" / "lightcurves_reports" / "epsf").glob("*_epsf.txt"))
    assert [p.name for p in out] == ["V1_Cyg_20261002_g_120_4_epsf.txt"]
    assert "# observer_code=UMIA" in out[0].read_text(encoding="utf-8")
