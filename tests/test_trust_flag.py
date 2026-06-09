"""Unit tests for trust_flag_core (Findings A+B, Phase E)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from trust_flag_core import (
    CompTrustThresholds,
    _UNEVALUATED_REASON,
    _UNEVALUATED_TRUST,
    check_star_scatter,
    classify_warnings,
    evaluate_target,
    format_export_trust_note,
    format_varastro_trust_comment,
    trust_level,
    write_trust_artifacts,
)

_TH = CompTrustThresholds.from_bounds(3, 8)


def test_unevaluated_defaults_red(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    phot = tmp_path / "photometry"
    phot.mkdir()
    summ = phot / "photometry_summary.csv"
    pd.DataFrame(
        [
            {"catalog_id": "486430957815961344", "vsx_name": "V842 Her", "n_clean": 5},
            {"catalog_id": "", "vsx_name": "BAD", "n_clean": 0},
        ]
    ).to_csv(summ, index=False)

    result = {
        "per_target": {
            "486430957815961344": {
                "trust": "GREEN",
                "trust_reason": "ok",
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
    }

    caplog.set_level("WARNING")
    write_trust_artifacts(result, photometry_dir=phot, write_per_target_json=False)

    out = pd.read_csv(summ, dtype={"catalog_id": str})
    bad_row = out[out["vsx_name"] == "BAD"].iloc[0]
    assert str(bad_row["trust"]) == _UNEVALUATED_TRUST
    assert str(bad_row["trust_reason"]) == _UNEVALUATED_REASON
    assert any("absent from trust map" in r.message for r in caplog.records)


def test_missing_check_star_adds_soft() -> None:
    hard, soft = classify_warnings(
        n_clean=_TH.strong,
        check_scatter=float("nan"),
        lc_quality="good",
        thresholds=_TH,
    )
    assert "no check-star verification available" in soft
    assert not hard
    assert trust_level(_TH.strong, hard, soft, _TH) == "YELLOW"


def test_present_clean_check_is_green() -> None:
    info = evaluate_target(
        catalog_id="123",
        vsx_name="T",
        n_clean=_TH.strong,
        lc_quality="good",
        check_scatter=0.005,
        thresholds=_TH,
    )
    assert info["trust"] == "GREEN"
    assert info["n_soft"] == 0
    assert info["n_hard"] == 0


def test_finite_check_thresholds() -> None:
    hard_hi, soft_hi = classify_warnings(
        n_clean=_TH.strong,
        check_scatter=0.06,
        lc_quality="good",
        thresholds=_TH,
    )
    assert any("high" in h for h in hard_hi)
    assert not soft_hi
    assert trust_level(_TH.strong, hard_hi, soft_hi, _TH) == "RED"

    hard_lo, soft_lo = classify_warnings(
        n_clean=_TH.strong,
        check_scatter=0.03,
        lc_quality="good",
        thresholds=_TH,
    )
    assert not hard_lo
    assert any("0.030" in s for s in soft_lo)
    assert trust_level(_TH.strong, hard_lo, soft_lo, _TH) == "YELLOW"


def test_max_two_soft_keeps_yellow() -> None:
    nc = _TH.min_comps + 1
    assert _TH.min_comps <= nc < _TH.strong
    hard, soft = classify_warnings(
        n_clean=nc,
        check_scatter=float("nan"),
        lc_quality="good",
        thresholds=_TH,
    )
    assert len(soft) == 2
    assert trust_level(nc, hard, soft, _TH) == "YELLOW"


def test_hard_forces_red() -> None:
    for n_clean, lq, chk in (
        (_TH.min_comps - 1, "good", 0.005),
        (_TH.strong, "saturated", 0.005),
        (_TH.strong, "good", 0.06),
    ):
        info = evaluate_target(
            catalog_id="x",
            vsx_name="",
            n_clean=n_clean,
            lc_quality=lq,
            check_scatter=chk,
            thresholds=_TH,
        )
        assert info["trust"] == "RED", (n_clean, lq, chk)


def test_format_notes_default_red() -> None:
    assert format_export_trust_note("", "").startswith("trust=RED")
    assert "RED" in format_varastro_trust_comment("", "")


def test_check_scatter_ddof(tmp_path: Path) -> None:
    lc = tmp_path / "lightcurves"
    lc.mkdir()
    cid = "999"
    pd.DataFrame({"kmag": [10.00, 10.02]}).to_csv(lc / f"check_kmag_{cid}.csv", index=False)
    scatter = check_star_scatter(tmp_path, cid)
    assert scatter == pytest.approx(0.01000, abs=1e-5)


def test_trust_json_written(tmp_path: Path) -> None:
    phot = tmp_path / "photometry"
    lc = phot / "lightcurves"
    lc.mkdir(parents=True)
    result = {
        "per_target": {
            "111": {
                "trust": "YELLOW",
                "trust_reason": "soft: no check-star verification available",
                "n_hard": 0,
                "n_soft": 1,
                "hard_warnings": [],
                "soft_warnings": ["no check-star verification available"],
                "n_clean": 5,
                "lc_quality": "good",
                "check_scatter": None,
                "min_comps": 3,
                "strong_comps": 5,
            }
        }
    }
    paths = write_trust_artifacts(result, photometry_dir=phot, update_summary=False)
    assert any(p.name == "trust_111.json" for p in paths)
    payload = json.loads((lc / "trust_111.json").read_text(encoding="utf-8"))
    assert payload["trust"] == "YELLOW"
