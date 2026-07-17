"""V3e ePSF FWHM QC estimator harness tests."""
from __future__ import annotations

from pathlib import Path

from tests.validation.v3e_epsf_fwhm import (
    PASS_RATIO_HI,
    PASS_RATIO_LO,
    run_v3e_epsf_fwhm,
    write_v3e_report,
)


def test_v3e_epsf_fwhm_pass():
    result = run_v3e_epsf_fwhm()
    assert result["status"] == "PASS"
    for row in result["cases"]:
        assert row["pass_new"]
        assert PASS_RATIO_LO <= row["ratio_new"] <= PASS_RATIO_HI
        assert abs(row["ratio_new"] - 1.0) < abs(row["ratio_old"] - 1.0)


def test_v3e_report_roundtrip(tmp_path: Path):
    result = run_v3e_epsf_fwhm()
    jp, mp = write_v3e_report(tmp_path, result)
    assert jp.is_file()
    assert mp.is_file()
    assert "OLD" in mp.read_text(encoding="ascii")
