"""Trust / check-star hardening (VYVAR_TRUST_CHECKSTAR_HARDENING_SPEC)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trust_flag_core import (
    CompTrustThresholds,
    _UNEVALUATED_REASON,
    _UNEVALUATED_TRUST,
    check_star_scatter,
    classify_warnings,
    evaluate_target,
    trust_level,
    write_trust_artifacts,
)

_TH = CompTrustThresholds.from_bounds(3, 8)
_MIN_CHK = 5


@pytest.mark.parametrize("trust_value", [None, ""])
def test_trust_map_missing_verdict_defaults_red(tmp_path: Path, trust_value) -> None:
    phot = tmp_path / "photometry"
    phot.mkdir()
    summ = phot / "photometry_summary.csv"
    pd.DataFrame([{"catalog_id": "111", "vsx_name": "T", "n_clean": 5}]).to_csv(summ, index=False)

    info = {
        "trust": trust_value,
        "trust_reason": None,
        "n_hard": 0,
        "n_soft": 0,
        "hard_warnings": [],
        "soft_warnings": [],
        "n_clean": 5,
        "lc_quality": "good",
        "check_scatter": 0.01,
        "min_comps": 3,
        "strong_comps": 5,
    }
    write_trust_artifacts(
        {"per_target": {"111": info}},
        photometry_dir=phot,
        write_per_target_json=False,
    )
    out = pd.read_csv(summ, dtype={"catalog_id": str})
    assert str(out.iloc[0]["trust"]) == _UNEVALUATED_TRUST
    assert str(out.iloc[0]["trust_reason"]) == _UNEVALUATED_REASON


def test_trust_map_absent_trust_key_defaults_red(tmp_path: Path) -> None:
    phot = tmp_path / "photometry"
    phot.mkdir()
    summ = phot / "photometry_summary.csv"
    pd.DataFrame([{"catalog_id": "112", "vsx_name": "T", "n_clean": 5}]).to_csv(summ, index=False)
    write_trust_artifacts(
        {
            "per_target": {
                "112": {
                    "n_hard": 0,
                    "n_soft": 0,
                    "hard_warnings": [],
                    "soft_warnings": [],
                    "n_clean": 5,
                    "lc_quality": "good",
                    "check_scatter": 0.01,
                    "min_comps": 3,
                    "strong_comps": 5,
                }
            }
        },
        photometry_dir=phot,
        write_per_target_json=False,
    )
    out = pd.read_csv(summ, dtype={"catalog_id": str})
    assert str(out.iloc[0]["trust"]) == _UNEVALUATED_TRUST


def test_trust_map_preserves_real_verdict(tmp_path: Path) -> None:
    phot = tmp_path / "photometry"
    phot.mkdir()
    summ = phot / "photometry_summary.csv"
    pd.DataFrame([{"catalog_id": "222", "vsx_name": "T", "n_clean": 5}]).to_csv(summ, index=False)
    write_trust_artifacts(
        {
            "per_target": {
                "222": {
                    "trust": "YELLOW",
                    "trust_reason": "review",
                    "n_hard": 0,
                    "n_soft": 1,
                    "hard_warnings": [],
                    "soft_warnings": ["x"],
                    "n_clean": 5,
                    "lc_quality": "good",
                    "check_scatter": None,
                    "min_comps": 3,
                    "strong_comps": 5,
                }
            }
        },
        photometry_dir=phot,
        write_per_target_json=False,
    )
    out = pd.read_csv(summ, dtype={"catalog_id": str})
    assert str(out.iloc[0]["trust"]) == "YELLOW"


@pytest.mark.parametrize("n_epochs", [2, 3, 4])
def test_thin_check_epochs_soft_not_green(n_epochs: int) -> None:
    hard, soft = classify_warnings(
        n_clean=_TH.strong,
        check_scatter=0.001,
        lc_quality="good",
        thresholds=_TH,
        n_check=n_epochs,
        check_min_epochs=_MIN_CHK,
    )
    assert any("insufficient check-star verification" in s for s in soft)
    assert not hard
    assert trust_level(_TH.strong, hard, soft, _TH) == "YELLOW"


def test_sufficient_check_low_scatter_green() -> None:
    info = evaluate_target(
        catalog_id="1",
        vsx_name="",
        n_clean=_TH.strong,
        lc_quality="good",
        check_scatter=0.005,
        thresholds=_TH,
        n_check=_MIN_CHK,
        check_min_epochs=_MIN_CHK,
    )
    assert info["trust"] == "GREEN"


def test_sufficient_check_high_scatter_red() -> None:
    info = evaluate_target(
        catalog_id="1",
        vsx_name="",
        n_clean=_TH.strong,
        lc_quality="good",
        check_scatter=0.06,
        thresholds=_TH,
        n_check=_MIN_CHK,
        check_min_epochs=_MIN_CHK,
    )
    assert info["trust"] == "RED"


def test_missing_check_file_soft() -> None:
    hard, soft = classify_warnings(
        n_clean=_TH.strong,
        check_scatter=float("nan"),
        lc_quality="good",
        thresholds=_TH,
        n_check=0,
        check_min_epochs=_MIN_CHK,
    )
    assert "no check-star verification available" in soft
    assert trust_level(_TH.strong, hard, soft, _TH) == "YELLOW"


def test_check_scatter_uses_sample_std_ddof1(tmp_path: Path) -> None:
    lc = tmp_path / "lightcurves"
    lc.mkdir()
    cid = "999"
    vals = [10.0, 10.2, 10.4, 10.1, 10.3]
    pd.DataFrame({"kmag": vals}).to_csv(lc / f"check_kmag_{cid}.csv", index=False)
    scatter, n = check_star_scatter(tmp_path, cid)
    assert n == len(vals)
    assert scatter == pytest.approx(float(np.nanstd(vals, ddof=1)), rel=0, abs=1e-9)


def test_short_baseline_thin_comp_thin_check_stays_yellow() -> None:
    nc = _TH.min_comps + 1
    hard, soft = classify_warnings(
        n_clean=nc,
        check_scatter=0.001,
        lc_quality="short_baseline",
        thresholds=_TH,
        n_frames=12,
        n_check=4,
        check_min_epochs=_MIN_CHK,
    )
    assert not hard
    assert any(s.startswith("short baseline") for s in soft)
    assert any("insufficient check-star verification" in s for s in soft)
    assert any("thin comp set" in s for s in soft)
    assert trust_level(nc, hard, soft, _TH) == "YELLOW"
