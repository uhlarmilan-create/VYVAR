# -*- coding: ascii -*-
"""OSC-3: band mapping, Gaia->Johnson comps, export gates."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from export_reports import (
    _prepare_osc_comp_df_for_export,
    resolve_aavso_filt_from_obs_group,
)
from gaia_johnson import (
    BPRP_MAX,
    BPRP_MIN,
    G_MAG_MAX,
    G_MAG_MIN,
    RIELLO_EDR3_FALLBACK,
    RODRIGUEZ2025_LANDOLT,
    transform_gaia_to_johnson,
)
from invariants_runtime import InvariantViolation, check_osc03_export_eligibility
from k2_extinction import K2_NONE_TOKENS, filter_token_from_obs_group
from osc_align import is_onerggb_internal_obs_group, obs_group_band_token


@pytest.mark.parametrize(
    ("obs_group", "expected"),
    [
        ("NoFilter_60_2_R", "TR"),
        ("NoFilter_60_2_G", "TG"),
        ("NoFilter_60_2_B", "TB"),
        ("NoFilter_60_2_oneRGGB", "CLEAR"),
        ("NoFilter_60_2", "NoFilter"),
    ],
)
def test_resolve_aavso_filt_from_obs_group(obs_group: str, expected: str) -> None:
    code, warn = resolve_aavso_filt_from_obs_group(obs_group)
    if expected == "CLEAR":
        assert code == ""
        assert warn is not None
    elif expected == "NoFilter":
        assert code in ("CV", "UNKN")
    else:
        assert code == expected
        assert warn is None


def test_onerggb_not_export_eligible() -> None:
    assert is_onerggb_internal_obs_group("NoFilter_60_2_oneRGGB")
    assert not is_onerggb_internal_obs_group("NoFilter_60_2_G")


def test_k2_none_tokens_tr_tg_tb() -> None:
    for og in ("NoFilter_60_2_R", "NoFilter_60_2_G", "NoFilter_60_2_B"):
        tok = filter_token_from_obs_group(og)
        assert tok in K2_NONE_TOKENS
        assert tok == obs_group_band_token(og)


def test_gaia_johnson_transform_roundtrip_grid() -> None:
    for bprp in [-0.2, 0.5, 1.2, 2.5, 4.0]:
        for g in [9.0, 12.0, 15.0]:
            for band in ("V", "B", "RC"):
                res = transform_gaia_to_johnson(g, bprp, band)
                assert res.ok
                assert math.isfinite(res.johnson_mag)
                assert res.johnson_mag_err >= 0.03


def test_gaia_johnson_out_of_validity_excluded() -> None:
    res = transform_gaia_to_johnson(12.0, BPRP_MAX + 0.5, "V")
    assert not res.ok
    assert "outside" in res.reason
    res2 = transform_gaia_to_johnson(G_MAG_MAX + 1.0, 1.0, "V")
    assert not res2.ok


def test_gaia_johnson_riello_fallback_differs_in_tests() -> None:
    g, bprp = 12.0, 1.0
    primary = transform_gaia_to_johnson(g, bprp, "V")
    fallback = transform_gaia_to_johnson(
        g, bprp, "V", coeff_set=RIELLO_EDR3_FALLBACK, g_min=G_MAG_MIN, g_max=G_MAG_MAX, bprp_min=BPRP_MIN, bprp_max=BPRP_MAX
    )
    assert primary.ok and fallback.ok
    assert primary.johnson_mag != pytest.approx(fallback.johnson_mag, abs=1e-6)


def test_osc_comp_johnson_plumbing() -> None:
    comp_df = pd.DataFrame(
        [
            {"catalog_id": "G001", "mag": 11.5, "bp_rp": 0.9},
            {"catalog_id": "G002", "mag": 17.0, "bp_rp": 0.8},
        ]
    )
    out, notes = _prepare_osc_comp_df_for_export(comp_df, "TG")
    assert len(out) == 2
    assert bool(out.iloc[0]["johnson_ok"])
    assert not bool(out.iloc[1]["johnson_ok"])
    assert notes


def test_osc03_gate_onerggb_fails() -> None:
    with pytest.raises(InvariantViolation) as exc:
        check_osc03_export_eligibility("NoFilter_60_2_oneRGGB", "CLEAR")
    assert exc.value.inv_id == "OSC-03"


def test_osc03_gate_rgb_pass() -> None:
    check_osc03_export_eligibility("NoFilter_60_2_G", "TG", meta={"invariants": []})


def test_osc03_gate_wrong_filt_fails() -> None:
    with pytest.raises(InvariantViolation):
        check_osc03_export_eligibility("NoFilter_60_2_G", "TR", meta={"invariants": []})


def test_methods_matrix_osc_rows() -> None:
    from citations import RunCitationContext, build_methods_matrix_lines

    ctx = RunCitationContext(
        osc_channel_export=True,
        osc_channel_binning=2,
        osc_transform_citation="Gaia DR3 CU5 Table 5.9",
    )
    lines = build_methods_matrix_lines(ctx)
    joined = "\n".join(lines)
    assert "OSC channel extraction: ON" in joined
    assert "OSC channel binning: 2x2 average" in joined
    assert "OSC Gaia->Johnson comps: ON" in joined


def test_export_onerggb_skipped(tmp_path: Path) -> None:
    from export_reports import export_lightcurve_reports

    phot = tmp_path / "photometry"
    phot.mkdir()
    lc = pd.DataFrame({"bjd": [2_459_000.0], "mag_calib": [12.0], "err": [0.01], "airmass": [1.1], "time_base": ["BJD_TDB"]})
    target = pd.Series({"catalog_id": "T1", "vsx_name": "TEST", "vsx_type": "EA", "mag": 12.0, "bp_rp": 0.5})
    summary = pd.Series({"obs_group": "NoFilter_60_2_oneRGGB"})
    paths = export_lightcurve_reports(
        phot / "lightcurves_reports",
        target,
        lc,
        pd.DataFrame(),
        summary,
        obs_group="NoFilter_60_2_oneRGGB",
    )
    assert paths == {}


def test_export_channel_filt_in_aavso_file(tmp_path: Path) -> None:
    from export_reports import export_lightcurve_reports

    phot = tmp_path / "photometry"
    (phot / "lightcurves_reports" / "aavso").mkdir(parents=True)
    lc = pd.DataFrame(
        {
            "bjd": [2_459_000.0],
            "mag_calib": [12.0],
            "err": [0.01],
            "airmass": [1.1],
            "source_file": ["f.csv"],
            "time_base": ["BJD_TDB"],
        }
    )
    target = pd.Series({"catalog_id": "T1", "vsx_name": "TEST", "vsx_type": "EA", "mag": 12.0, "bp_rp": 0.5})
    summary = pd.Series({"obs_group": "NoFilter_60_2_G", "n_frames": 1, "n_good_comp": 3})
    comp = pd.DataFrame([{"catalog_id": "C1", "mag": 11.0, "bp_rp": 0.8, "p2p_rms": 0.01, "w_rel": 1.0, "tier": 1}])
    paths = export_lightcurve_reports(
        phot / "lightcurves_reports",
        target,
        lc,
        comp,
        summary,
        obs_group="NoFilter_60_2_G",
    )
    assert "aavso" in paths
    text = paths["aavso"].read_text(encoding="utf-8")
    assert ",TG," in text or text.split("\n")[-2].split(",")[4] == "TG"
    assert "oneRGGB" not in text
    assert "Gaia DR3 CU5 Table 5.9" in text
