"""VAR-TAB-2026-10: interactive Variability tab, TESS folder, period / phase helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tess_runner import tess_dir_for
from variability_store import (
    epoch_of_minimum,
    load_adopted_periods,
    pdm_theta,
    phase_of,
    phased_text,
    save_adopted_period,
)

SRC = Path(__file__).resolve().parents[2] / "src_py"


def test_tess_folder_is_TESS(tmp_path: Path) -> None:
    assert tess_dir_for(tmp_path / "photometry", " 123 ") == tmp_path / "photometry" / "TESS" / "123"


def _sine(period: float = 0.4647, n: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    t = 2459000.0 + np.sort(rng.uniform(0, 27, n))
    m = 0.15 * np.cos(2 * np.pi * (t - 2459000.1) / period) + rng.normal(0, 0.01, n)
    return t, m


def test_pdm_theta_minimum_at_true_period_and_epoch() -> None:
    t, m = _sine()
    good = pdm_theta(phase_of(t, 0.4647, t[0]), m)
    off = pdm_theta(phase_of(t, 0.4647 * 1.02, t[0]), m)
    assert good < 0.05 < 0.5 < off
    t0 = epoch_of_minimum(t, m, 0.4647)
    # dmag = +0.15 cos(...): the star is faintest (largest dmag) at 2459000.1
    ph = phase_of(np.array([2459000.1]), 0.4647, t0)[0]
    assert min(ph, 1 - ph) < 0.03


def test_adopted_period_roundtrip_and_phased_export(tmp_path: Path) -> None:
    save_adopted_period(tmp_path, "42", 0.4647021, 2459000.25)
    save_adopted_period(tmp_path, "43", 1.5, 2459001.0)
    (tmp_path / "variability_periods.json").write_text(
        json.dumps({**json.loads((tmp_path / "variability_periods.json").read_text()), "bad": {"period_d": "x"}}))
    ap = load_adopted_periods(tmp_path)
    assert set(ap) == {"42", "43"} and ap["42"]["period_d"] == 0.4647021
    txt = phased_text(np.array([2459000.25, 2459000.5]), np.array([0.1, np.nan]), np.array([0.01, 0.01]), 0.5, 2459000.25, "X")
    lines = txt.splitlines()
    assert lines[0].startswith("# X  P = 0.5000000 d") and lines[2].startswith("0.00000 0.1000") and len(lines) == 3


def test_noise_curves_and_star_classes() -> None:
    from ui_variability import star_class
    from variability_plots import noise_curves

    rng = np.random.default_rng(3)
    n = 400
    mag = rng.uniform(9, 14, n)
    e = 0.002 * 10 ** (0.2 * (mag - 9))
    f = pd.DataFrame({"mag": mag, "e": e, "s": e * rng.uniform(0.8, 1.2, n), "gates_ok": True, "known_variable": False})
    mids, sexp, slim = noise_curves(f, {"noise_model": {"a": 1.0, "floor_mag": 0.003}, "z_threshold": 4.0})
    assert np.all(slim > sexp) and np.all(np.diff(mids) == 0.5)
    base = {"candidate": False, "known_detected": False, "known_variable": False, "target": False, "gates_ok": True, "systematic": False}
    assert star_class(pd.Series({**base, "candidate": True, "known_detected": True})) == "cand"
    assert star_class(pd.Series({**base, "known_detected": True})) == "kdet"
    assert star_class(pd.Series({**base, "target": True})) == "known"
    assert star_class(pd.Series({**base, "systematic": True})) == "sys"
    assert star_class(pd.Series({**base, "gates_ok": False, "systematic": True})) == "rej"
    assert star_class(pd.Series(base)) == "field"


def _draft(tmp_path: Path) -> Path:
    ps = tmp_path / "draft_000001" / "platesolve" / "NoFilter_60_2"
    phot = ps / "photometry"
    (phot / "lightcurves").mkdir(parents=True)
    rng = np.random.default_rng(5)
    n = 300
    mag = rng.uniform(9, 14, n)
    e = 0.002 * 10 ** (0.2 * (mag - 9))
    f = pd.DataFrame({"catalog_id": [str(1000 + i) for i in range(n)], "mag": mag, "e": e, "s": e, "bp_rp": 0.8,
                      "z_s": 0.0, "z_eta": 0.0, "s_exp": e, "amplitude_mag": 0.01, "ra_deg": 200.0, "dec_deg": 40.0,
                      "gates_ok": True, "known_variable": False, "target": False, "systematic": False,
                      "candidate": False, "known_detected": False, "index": ""})
    f.loc[0, ["candidate", "s", "z_s", "index"]] = [True, 0.3, 8.0, "scatter"]
    f.loc[1, ["known_variable", "known_detected", "s", "z_s", "index"]] = [True, True, 0.2, 6.0, "scatter"]
    f["label"], f["catalogs"] = "", ""
    f.loc[0, ["label", "catalogs"]] = ["V1", "none of the catalogues lists it"]
    f.to_csv(phot / "variability_field.csv", index=False)  # OUTPUT-REVIEW-2026-10: one file
    pd.DataFrame({"catalog_id": ["1001"], "vsx_name": ["XX Cnv"], "vsx_type": ["EW"], "vsx_period": [0.4647]}).to_csv(
        ps / "variable_targets.csv", index=False)
    (phot / "variability_meta.json").write_text(json.dumps({"status": "ok", "n_stars": n, "n_frames": 50, "n_gates_ok": n,
                                                             "noise_model": {"a": 1.0, "floor_mag": 0.003}, "z_threshold": 4.0,
                                                             "n_candidates": 1, "n_known_detected": 1, "time_column": "bjd_tdb_mid"}))
    t, m = _sine()
    pd.DataFrame({"time": t[:60], "dmag": m[:60], "err": 0.01, "fwhm": 2.0, "airmass": 1.1}).to_csv(
        phot / "lightcurves" / "varcand_1001.csv", index=False)
    tdir = tess_dir_for(phot, "1001")
    tdir.mkdir(parents=True)
    pd.DataFrame({"bjd_tdb": t, "dmag": m, "err": 0.01}).to_csv(tdir / "tess_lc.csv", index=False)
    fr = np.linspace(0.05, 10, 4000)
    pd.DataFrame({"frequency_per_d": fr, "ls_power": np.exp(-((fr - 1 / 0.4647) ** 2) / 0.001), "aov_theta": 0.0}).to_csv(
        tdir / "periodogram.csv", index=False)
    (tdir / "tess_result.json").write_text(json.dumps({"verdict": "RELIABLE", "best": {"period_d": 0.4647, "method": "test"},
                                                        "sectors": [{"sector": 1, "cadence_s": 120, "n_points": len(t)}]}))
    return tmp_path / "draft_000001"


def test_tab_renders_and_period_controls(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    draft = _draft(tmp_path)
    app = tmp_path / "app.py"
    app.write_text(
        "import sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "from config import AppConfig\nfrom ui_variability import render_variability_dashboard\n"
        f"render_variability_dashboard(AppConfig(), draft_dir=Path({str(draft)!r}))\n")
    at = AppTest.from_file(str(app), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert [h.value for h in at.subheader] == ["Hockey-stick: scatter of every star vs brightness", "Variable stars in this field"]
    assert "NEW V1" in at.markdown[0].value  # candidates first
    at.session_state["var_sel_NoFilter_60_2"] = "1001"
    at.run()
    assert not at.exception, at.exception
    assert "XX Cnv" in at.markdown[0].value and "EW" in at.markdown[0].value
    pin = next(n for n in at.number_input if n.key == "var_pin_1001")
    assert abs(pin.value - 0.4647) < 1e-9
    theta0 = float(next(m for m in at.metric if m.label.startswith("\u0398")).value)
    at.button(key="var_p10_1001").click().run()
    pin = next(n for n in at.number_input if n.key == "var_pin_1001")
    assert abs(pin.value - (0.4647 + 10e-4)) < 1e-9
    assert float(next(m for m in at.metric if m.label.startswith("\u0398")).value) > theta0
    at.button(key="var_dbl_1001").click().run()
    assert abs(next(n for n in at.number_input if n.key == "var_pin_1001").value - 2 * (0.4647 + 10e-4)) < 1e-9
    at.button(key="var_ptess_1001").click().run()
    assert abs(next(n for n in at.number_input if n.key == "var_pin_1001").value - 0.4647) < 1e-9
    assert not at.exception, at.exception


def test_app_and_report_do_not_read_old_tess_folder() -> None:
    for name in ("ui_variability.py", "photometry_report.py", "tess_runner.py", "variability_store.py"):
        assert '"_tess"' not in (SRC / name).read_text(encoding="utf-8"), name
    assert "import streamlit" not in (SRC / "variability_store.py").read_text(encoding="utf-8")
