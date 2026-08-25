# -*- coding: ascii -*-
"""D3: comparison candidacy before RMS ceiling."""

from __future__ import annotations

import pandas as pd
import pytest

from d3_comparison_candidacy import D3CandidacyError, apply_d3_comparison_candidacy


def _row(*, state: str, gate: str, resid: float, snr: float, rms: float, cid: str) -> dict:
    return {
        "source_state": state,
        "vy_identity_gate": gate,
        "gaia_dao_resid_px": resid,
        "snr": snr,
        "comp_rms": rms,
        "catalog_id": cid,
        "name": cid,
        "is_saturated": False,
        "vsx_known_variable": False,
        "likely_saturated": False,
    }


def test_d3_only_honest_star_is_candidate() -> None:
    df = pd.DataFrame(
        [
            _row(state="DETECTED_P1", gate="fail", resid=59.0, rms=0.02, snr=80.0, cid="ghost"),
            _row(state="DETECTED_P1", gate="ok", resid=0.8, rms=0.07, snr=40.0, cid="honest"),
            _row(
                state="catalog_membership",
                gate="ok",
                resid=0.5,
                rms=0.03,
                snr=30.0,
                cid="inject",
            ),
        ]
    )
    mask, meta = apply_d3_comparison_candidacy(df, fwhm_dao_px=1.25, solve_rms_px=1.44, log_label="t")
    out = df.loc[mask]
    assert list(out["catalog_id"]) == ["honest"]
    assert int(meta["n_out"]) == 1
    assert int(meta["drops"]["source_state"]) == 1


def test_d3_missing_column_raises_name() -> None:
    df = pd.DataFrame(
        [{"source_state": "DETECTED_P1", "vy_identity_gate": "ok", "snr": 20.0}]
    )
    with pytest.raises(D3CandidacyError, match="gaia_dao_resid_px"):
        apply_d3_comparison_candidacy(df, fwhm_dao_px=2.5, solve_rms_px=1.3)
