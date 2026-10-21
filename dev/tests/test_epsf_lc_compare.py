"""EPSF-LC-COMPARE-2026-10: aperture vs PSF differential light-curve statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epsf_lc_compare import compare_draft, load_pair, p2p_scatter


def test_p2p_ignores_slow_variability() -> None:
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, 400)
    noise = rng.normal(0, 0.01, t.size)
    assert p2p_scatter(noise) == pytest.approx(0.01, rel=0.12)
    # a 0.5 mag eclipse-like trend barely changes the epoch-to-epoch scatter
    assert p2p_scatter(noise + 0.5 * np.sin(2 * np.pi * t)) == pytest.approx(0.01, rel=0.15)
    assert np.isnan(p2p_scatter([0.1, 0.2]))


def _draft(tmp: Path, *, slope_mmag_px: float) -> Path:
    ps = tmp / "platesolve" / "G"
    lc = ps / "photometry" / "lightcurves"
    lc.mkdir(parents=True)
    rng = np.random.default_rng(3)
    n = 120
    frames = [f"F_{i:03d}" for i in range(n)]
    fwhm = 2.0 + 0.5 * np.sin(np.linspace(0, 6, n))
    signal = 0.3 * np.sin(np.linspace(0, 1, n))
    ap = signal + rng.normal(0, 0.008, n)
    psf = signal + rng.normal(0, 0.006, n) + 0.02 + 1e-3 * slope_mmag_px * (fwhm - 2.0)
    flag = ["normal"] * n
    flag[5] = "saturated"
    flag[6] = "low_snr"  # a measurement: stays in
    pd.DataFrame(
        {"bjd": 2460000 + np.arange(n) / 100, "source_file": [f"proc_{f}.csv" for f in frames],
         "delta_mag": ap, "err": 0.008, "flag": flag}
    ).to_csv(lc / "lightcurve_1.csv", index=False)
    ok = np.ones(n, dtype=bool)
    ok[7] = False
    with (lc / "lightcurve_1_psf.csv").open("w") as fh:
        fh.write("# INTERNAL\n")
        pd.DataFrame(
            {"bjd": 2460000 + np.arange(n) / 100, "source_file": [f"proc_{f}.csv" for f in frames], "psf_fit_ok": ok,
             "psf_delta_mag": np.where(ok, psf, np.nan), "psf_delta_mag_err": 0.006}
        ).to_csv(fh, index=False)
    man = ps / "epsf_frames"
    man.mkdir()
    (man / "epsf_frames_manifest.json").write_text(
        json.dumps({"frames": [{"frame": f + ".fits", "fwhm_px": float(w)} for f, w in zip(frames, fwhm)]})
    )
    return ps


def test_pair_uses_only_epochs_both_methods_kept_and_finds_the_seeing_slope(tmp_path: Path) -> None:
    ps = _draft(tmp_path, slope_mmag_px=12.0)
    df = load_pair(ps / "photometry" / "lightcurves", "1")
    assert int(df["ap_ok"].sum()) == 119 and int(df["psf_ok"].sum()) == 119
    t = compare_draft(ps, ["1", "2"])  # star 2 has no curves -> skipped
    assert list(t["catalog_id"]) == ["1"]
    r = t.iloc[0]
    assert r["n_both"] == 118
    assert r["diff_median_mmag"] == pytest.approx(20.0, abs=3.0)
    assert r["diff_fwhm_slope_mmag_px"] == pytest.approx(12.0, abs=4.0)
    assert r["ap_p2p_over_err"] == pytest.approx(1.0, abs=0.2)
    assert r["psf_p2p_mmag"] < r["ap_p2p_mmag"]


def test_missing_psf_curve_raises(tmp_path: Path) -> None:
    ps = _draft(tmp_path, slope_mmag_px=0.0)
    (ps / "photometry" / "lightcurves" / "lightcurve_1_psf.csv").unlink()
    with pytest.raises(FileNotFoundError):
        load_pair(ps / "photometry" / "lightcurves", "1")


def test_aperture_error_is_converted_to_magnitudes(tmp_path: Path) -> None:
    """The aperture light curve's ``err`` is relative flux; load_pair gives both errors in magnitudes."""
    ps = _draft(tmp_path, slope_mmag_px=0.0)
    df = load_pair(ps / "photometry" / "lightcurves", "1")
    assert df["ap_err"].iloc[0] == pytest.approx(0.008 * 2.5 / np.log(10.0))


def test_epsf_page_data_and_figure(tmp_path: Path) -> None:
    """EPSF-PDF-2026-10: facts of the ePSF page (epoch accounting from the PSF header, merged neighbour
    from the manifest) and the two-panel figure."""
    import star_page_m3 as m3

    ps = _draft(tmp_path, slope_mmag_px=0.0)
    lc = ps / "photometry" / "lightcurves"
    p = lc / "lightcurve_1_psf.csv"
    p.write_text("# psf_lc_n_epochs_target_saturated=2\n# ensemble_n_comp=6\n" + p.read_text())
    man = ps / "epsf_frames" / "epsf_frames_manifest.json"
    m = json.loads(man.read_text())
    m["fit_sources"] = {"merged_neighbours": [{"catalog_id": "1", "neighbour_id": "9", "sep_px": 2.0, "dmag": 3.75}]}
    m["fit_set"] = {"selection": "EA, EW"}
    man.write_text(json.dumps(m))
    d = m3.epsf_page_data(lc, ps, "1")
    facts = dict(d["facts"])
    assert facts["ePSF epochs"].startswith("119 of 120") and "dropped: 2 with the target saturated" in facts["ePSF epochs"]
    assert facts["Comparison stars"].startswith("6")
    assert "3.2 % of the light" in facts["Unresolved neighbour"]
    assert facts["Targets of the run"] == "EA, EW"
    assert m3.plot_lc_epsf(d["df"], tmp_path / "e.png").is_file()
    assert m3.epsf_page_data(lc, ps, "2") is None  # no ePSF curve -> no page
