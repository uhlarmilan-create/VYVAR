"""GATE-REGIME-01: regime exclusivity + INV-NO-SILENT-EMPTY at derived admission."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))

from comp_pool_noise import CompPoolAdmissionError, CompPoolRegime  # noqa: E402
from config import AppConfig  # noqa: E402
from invariants_runtime import PopulationEmptiedError, assert_population_nonempty  # noqa: E402
from photometry_core import build_global_comp_pool  # noqa: E402


def test_inv_no_silent_empty_fire_proof() -> None:
    with pytest.raises(PopulationEmptiedError, match="COMP_POOL_DERIVED_ADMIT") as ei:
        assert_population_nonempty(
            n_in=12,
            n_out=0,
            rule_id="COMP_POOL_DERIVED_ADMIT",
            threshold={"stability_excess_mad": 1.9},
            unit="mixed_derived",
            population="stars in global comp pool",
        )
    err = ei.value
    assert err.inv_id == "INV-NO-SILENT-EMPTY"
    assert err.n_in == 12
    assert "rule_id=COMP_POOL_DERIVED_ADMIT" in str(err)


def test_assert_population_nonempty_allows_passthrough() -> None:
    assert_population_nonempty(
        n_in=5,
        n_out=3,
        rule_id="X",
        threshold=0.1,
        unit="mag",
        population="comps",
    )
    assert_population_nonempty(
        n_in=0,
        n_out=0,
        rule_id="X",
        threshold=0.1,
        unit="mag",
        population="comps",
    )


def _static_pool_df(n: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "catalog_id": str(1000 + i),
                "name": str(1000 + i),
                "x": 100.0 + 10 * i,
                "y": 100.0 + 10 * i,
                "is_usable": True,
                "is_saturated": False,
                "is_noisy": False,
                "vsx_known_variable": False,
                "likely_saturated": False,
                "zone": "linear",
                "source_state": "DETECTED_P1",
                "vy_identity_gate": "ok",
                "gaia_dao_resid_px": 0.2,
                "snr": 80.0,
            }
        )
    return pd.DataFrame(rows)


def test_build_global_comp_pool_empty_derived_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Regression: real build_global_comp_pool raises when derived admits zero stars."""

    def _fake_analyze(*_a, **_k):
        dec = pd.DataFrame(
            {
                "catalog_id": ["1000", "1001", "1002", "1003", "1004"],
                "admit": [False, False, False, False, False],
                "reject_reasons": [
                    "fainter_than_12.0",
                    "mad_excess>1.9",
                    "inv_eta>0.8",
                    "dilution<0.99",
                    "detect_frac<1.0",
                ],
            }
        )
        return {
            "n_admitted": 0,
            "n_stars": 5,
            "decisions": dec,
            "thresholds": {"stability_excess_mad": 1.9},
            "fit": {"sigma_sys_mag": 0.01},
            "scint_vs_sys": {},
        }

    monkeypatch.setattr("comp_pool_noise.analyze_draft_comp_pool", _fake_analyze)
    monkeypatch.setattr(
        "photometry_core.compute_global_pool_rms_map",
        lambda **_k: {str(1000 + i): 0.01 for i in range(5)},
    )

    cfg = AppConfig()
    cfg.comp_pool_derived_admission = True
    art_dir = tmp_path / "photometry"
    art_dir.mkdir()
    with pytest.raises(PopulationEmptiedError, match="COMP_POOL_DERIVED_ADMIT"):
        build_global_comp_pool(
            masterstars_df=_static_pool_df(),
            per_frame_csv_paths=[tmp_path / "proc_001.csv"],
            csv_cache={},
            variable_target_catalog_ids=frozenset(),
            safe_bbox=None,
            chip_fw=1000,
            chip_fh=1000,
            chip_interior_margin_px=0,
            max_comp_rms=0.1,
            cfg=cfg,
            admission_artifact_dir=art_dir,
            photometry_dir_for_meta=art_dir,
        )
    art = art_dir / "comp_pool_admission.json"
    assert art.is_file()
    payload = json.loads(art.read_text(encoding="utf-8"))
    assert payload["regime"] == CompPoolRegime.DERIVED.value
    assert payload["rules"][0]["n_in"] == 5
    assert payload["rules"][0]["n_out"] == 0
    assert payload["reject_reason_counts"]


def test_build_global_comp_pool_failed_does_not_legacy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _boom(*_a, **_k):
        raise RuntimeError("synthetic derived failure")

    monkeypatch.setattr("comp_pool_noise.analyze_draft_comp_pool", _boom)
    cfg = AppConfig()
    cfg.comp_pool_derived_admission = True
    with pytest.raises(CompPoolAdmissionError, match="synthetic derived failure"):
        build_global_comp_pool(
            masterstars_df=_static_pool_df(),
            per_frame_csv_paths=[tmp_path / "proc_001.csv"],
            csv_cache={},
            variable_target_catalog_ids=frozenset(),
            safe_bbox=None,
            chip_fw=1000,
            chip_fh=1000,
            chip_interior_margin_px=0,
            max_comp_rms=0.1,
            cfg=cfg,
            admission_artifact_dir=tmp_path,
        )
