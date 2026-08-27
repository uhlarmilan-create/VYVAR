# -*- coding: ascii -*-
"""INV-EPSF-COMPLETE-01: all-dropped PSF LCs fail the night-run completeness gate."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from night_run import audit_photometry_completeness, audit_psf_lc_completeness


def _aperture_lc(path: Path, tid: str) -> None:
    path.write_text(
        "bjd,delta_mag,err\n2461154.3,0.1,0.01\n",
        encoding="ascii",
    )


def _dropped_psf_lc(path: Path) -> None:
    path.write_text(
        "# INTERNAL DIAGNOSTIC PRODUCT - NOT FOR AAVSO/VARASTRO SUBMISSION\n"
        "# psf_lc_n_epochs_full=0\n"
        "# psf_lc_n_epochs_dropped_pin=134\n"
        "bjd,psf_delta_mag,psf_epoch_drop_reason\n"
        "2461154.3,,comp_psf_fail:1497771992240531712\n",
        encoding="ascii",
    )


def _ok_psf_lc(path: Path) -> None:
    path.write_text(
        "# psf_lc_n_epochs_full=134\n"
        "# psf_lc_n_epochs_dropped_pin=0\n"
        "bjd,psf_delta_mag,psf_epoch_drop_reason\n"
        "2461154.3,0.01,\n",
        encoding="ascii",
    )


def test_all_dropped_psf_lcs_fail_completeness(tmp_path: Path) -> None:
    phot = tmp_path
    lc = phot / "lightcurves"
    lc.mkdir()
    pd.DataFrame({"catalog_id": ["T1", "T2"], "mag": [11.0, 12.0]}).to_csv(
        phot / "active_targets.csv", index=False
    )
    pd.DataFrame({"catalog_id": ["T1", "T2"], "lc_rms": [0.05, 0.04]}).to_csv(
        phot / "photometry_summary.csv", index=False
    )
    _aperture_lc(lc / "lightcurve_T1.csv", "T1")
    _aperture_lc(lc / "lightcurve_T2.csv", "T2")
    _dropped_psf_lc(lc / "lightcurve_T1_psf.csv")
    _dropped_psf_lc(lc / "lightcurve_T2_psf.csv")
    audit = audit_photometry_completeness(phot, require_psf=True)
    assert audit["ok"] is False
    assert "all" in str(audit.get("psf_error") or "").lower()
    psf = audit_psf_lc_completeness(phot, require=True)
    assert psf["ok"] is False
    assert psf["n_all_dropped"] == 2
    assert psf["n_full_positive"] == 0


def test_aperture_only_tree_skips_psf_audit(tmp_path: Path) -> None:
    phot = tmp_path
    pd.DataFrame({"catalog_id": ["T1"], "mag": [11.0]}).to_csv(
        phot / "active_targets.csv", index=False
    )
    pd.DataFrame({"catalog_id": ["T1"], "lc_rms": [0.05]}).to_csv(
        phot / "photometry_summary.csv", index=False
    )
    audit = audit_photometry_completeness(phot)
    assert audit["ok"] is True
    assert audit["psf"]["applicable"] is False


def test_mixed_n_full_passes(tmp_path: Path) -> None:
    phot = tmp_path
    lc = phot / "lightcurves"
    lc.mkdir()
    pd.DataFrame({"catalog_id": ["T1", "T2"], "mag": [11.0, 12.0]}).to_csv(
        phot / "active_targets.csv", index=False
    )
    pd.DataFrame({"catalog_id": ["T1", "T2"], "lc_rms": [0.05, 0.04]}).to_csv(
        phot / "photometry_summary.csv", index=False
    )
    _aperture_lc(lc / "lightcurve_T1.csv", "T1")
    _aperture_lc(lc / "lightcurve_T2.csv", "T2")
    _ok_psf_lc(lc / "lightcurve_T1_psf.csv")
    _dropped_psf_lc(lc / "lightcurve_T2_psf.csv")
    audit = audit_photometry_completeness(phot, require_psf=True)
    assert audit["ok"] is True
    assert audit["psf"]["n_full_positive"] == 1
    assert audit["psf"]["n_all_dropped"] == 1


def test_app_and_dashboard_call_run_epsf_stage() -> None:
    root = Path(__file__).resolve().parents[2]
    app = (root / "src_py" / "app.py").read_text(encoding="utf-8")
    i = app.index('pending.get("kind") == "run_epsf"')
    j = app.index("else:", i)
    body = app[i:j]
    assert "run_epsf_stage" in body
    assert "run_epsf_psf_merge_job" not in body
    dash = (root / "src_py" / "ui_epsf_dashboard.py").read_text(encoding="utf-8")
    k = dash.index("Write internal PSF light curves")
    chunk = dash[k : k + 2500]
    assert "run_epsf_stage" in chunk
    assert "write_internal_psf_lightcurves(" not in chunk
