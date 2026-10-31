"""EPSF-EXPORT-2026-10: ePSF light curve export for the user (time, delta_mag, err + header)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from epsf_export import find_epsf_export, write_epsf_user_lightcurves


def _draft(tmp_path: Path) -> Path:
    ps = tmp_path / "g_120_4"
    lc = ps / "photometry" / "lightcurves"
    lc.mkdir(parents=True)
    pd.DataFrame({"catalog_id": ["1", "2"], "vsx_name": ["V1 Cyg", "V2 Cyg"], "vsx_type": ["EW", "EA"],
                  "vsx_period": [0.3, 1.2], "ra_deg": [320.1, 320.2], "dec_deg": [51.7, 51.8], "mag": [13.2, 14.1],
                  "catalog": "VSX"}).to_csv(ps / "photometry" / "active_targets.csv", index=False)
    head = ("# epsf_model_sha256=abc\n# ensemble_n_comp=3\n# ensemble_pinned_ids=7,8,9\n"
            "# psf_lc_n_epochs_full=3\n# psf_lc_n_epochs_dropped_pin=1\n# psf_lc_n_epochs_target_saturated=0\n"
            "# psf_lc_time_base=BJD_TDB\n")
    rows = ("bjd,psf_fit_ok,psf_delta_mag,psf_delta_mag_err\n2461316.40,True,0.512,0.008\n"
            "2461316.30,True,0.501,0.007\n2461316.35,False,,\n2461316.45,True,0.523,0.009\n")
    (lc / "lightcurve_1_psf.csv").write_text(head + rows)
    (lc / "lightcurve_2_psf.csv").write_text(head + "bjd,psf_fit_ok,psf_delta_mag,psf_delta_mag_err\n2461316.3,False,,\n")
    (ps / "epsf_frames").mkdir()
    (ps / "epsf_frames" / "epsf_frames_manifest.json").write_text(json.dumps({
        "oversampling": 2, "frames": [{"fwhm_px": 4.0}, {"fwhm_px": 5.0}], "spatial": {"grid": [3, 3]},
        "fit_sources": {"merged_neighbours": [{"catalog_id": "1", "neighbour_id": "55", "sep_px": 1.8, "dmag": 3.2}]}}))
    return ps


def test_export_writes_kept_epochs_sorted_with_header(tmp_path: Path) -> None:
    ps = _draft(tmp_path)
    r = write_epsf_user_lightcurves(ps, ["1", "2", "3"], setup="g_120_4", observer_code="UMIA")
    assert r["n_written"] == 1 and r["skipped"] == {"2": "no epoch kept", "3": "no ePSF light curve"}
    p = Path(r["paths"][0])
    assert p.parent == ps / "photometry" / "lightcurves_reports" / "epsf"
    assert p.name.startswith("V1_Cyg_2026100") and p.name.endswith("_g_120_4_epsf.txt")
    text = p.read_text(encoding="utf-8").splitlines()
    hdr = dict(line[2:].split("=", 1) for line in text if line.startswith("# ") and "=" in line)
    assert hdr["star"] == "V1 Cyg" and hdr["catalog_id"] == "1" and hdr["observer_code"] == "UMIA"
    assert hdr["filter"] == "g" and hdr["exptime_s"] == "120" and hdr["ensemble_comp_ids"] == "7,8,9"
    assert hdr["n_epochs"] == "3" and hdr["epochs_dropped"].startswith("1 (frame not fitted")
    assert "3x3" in hdr["method"] and "epsf_targets_selection" not in hdr  # S6-FIX: epsf_targets removed
    assert hdr["unresolved_neighbour"].startswith("Gaia DR3 55 at 1.8 px")
    data = pd.read_csv(p, comment="#")
    assert list(data.columns) == ["bjd_tdb", "delta_mag", "err"]
    assert data["bjd_tdb"].is_monotonic_increasing and len(data) == 3
    assert data["delta_mag"].tolist() == [0.501, 0.512, 0.523]
    assert find_epsf_export(ps / "photometry", "1", "g_120_4") == p
    assert find_epsf_export(ps / "photometry", "1", "r_100_4") is None


def test_rerun_replaces_the_target_file(tmp_path: Path) -> None:
    ps = _draft(tmp_path)
    p = Path(write_epsf_user_lightcurves(ps, ["1"], setup="g_120_4")["paths"][0])
    (ps / "photometry" / "lightcurves" / "lightcurve_1_psf.csv").write_text(
        "bjd,psf_fit_ok,psf_delta_mag,psf_delta_mag_err\n2461316.3,False,,\n")
    r = write_epsf_user_lightcurves(ps, ["1"], setup="g_120_4")
    assert r["n_written"] == 0 and not p.exists()  # no stale file of an earlier run


def test_g09_39_jd_fallback_light_curve_is_not_exported_as_bjd(tmp_path: Path) -> None:
    ps = _draft(tmp_path)
    f = ps / "photometry" / "lightcurves" / "lightcurve_1_psf.csv"
    f.write_text(f.read_text().replace("psf_lc_time_base=BJD_TDB", "psf_lc_time_base=JD_FALLBACK"))
    r = write_epsf_user_lightcurves(ps, ["1"], setup="g_120_4")
    assert r["n_written"] == 0 and "BJD_TDB required" in r["skipped"]["1"]
