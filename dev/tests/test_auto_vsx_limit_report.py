"""AUTO-VSX-LIMIT: report-layer VSX limit vs field-depth check (pure + smoke)."""

from __future__ import annotations

from photometry_report import (
    load_field_depth_metrics,
    resolved_facts_model,
    vsx_limit_vs_depth_status,
)
from config import AppConfig


def test_vsx_depth_ok_within_margin() -> None:
    st = vsx_limit_vs_depth_status(14.5, g_lim_90=14.4, snr5=14.6)
    assert st["status"] == "ok"
    assert st["warn"] is False
    assert "14.5" in st["line"]
    assert "G_lim_90=14.40" in st["line"]
    assert "SNR5=14.60" in st["line"]


def test_vsx_depth_warn_when_limit_deeper() -> None:
    # limit 15.0 vs min(14.0, 14.5)=14.0 ? over by 1.0 > 0.3
    st = vsx_limit_vs_depth_status(15.0, g_lim_90=14.0, snr5=14.5)
    assert st["status"] == "warn"
    assert st["warn"] is True
    assert "deeper than measured field depth" in st["message"]


def test_vsx_depth_boundary_exactly_margin() -> None:
    # limit == min + 0.3 ? not warn (strict >)
    st = vsx_limit_vs_depth_status(14.3, g_lim_90=14.0, snr5=14.5)
    assert st["warn"] is False
    assert st["status"] == "ok"


def test_vsx_depth_na_when_depths_missing() -> None:
    st = vsx_limit_vs_depth_status(14.5, None, None)
    assert st["status"] == "n/a"
    assert st["warn"] is False
    assert "G_lim_90=n/a" in st["line"]
    assert "SNR5=n/a" in st["line"]


def test_vsx_depth_uses_min_of_available() -> None:
    # only SNR5 present; warn if limit deeper than SNR5+0.3
    st = vsx_limit_vs_depth_status(15.0, None, 14.0)
    assert st["warn"] is True
    assert st["depth_min"] == 14.0


def test_load_field_depth_missing_artifacts_no_crash() -> None:
    depth = load_field_depth_metrics({}, photometry_dir=None)
    assert depth["g_lim_90"] is None
    assert depth["snr5"] is None


def test_resolved_facts_tolerates_missing_depth() -> None:
    model = resolved_facts_model(
        {"g_lim_90": None, "resolved_facts": {"filter": "V", "exptime_s": 60.0}},
        photometry_dir=None,
        cfg=None,
    )
    assert any(r["label"] == "VSX auto-target scope" for r in model["rows"])
    vsx_row = next(r for r in model["rows"] if r["label"] == "VSX auto-target scope")
    assert "detection-limited" in vsx_row["value"]
    assert model["vsx_depth"]["status"] == "n/a"
    assert model["vsx_depth"]["warn"] is False


def test_resolved_facts_detection_limited_scope() -> None:
    model = resolved_facts_model(
        {"g_lim_90": 14.0, "resolved_facts": {"filter": "V"}},
        photometry_dir=None,
        cfg=AppConfig(),
    )
    vsx_row = next(r for r in model["rows"] if r["label"] == "VSX auto-target scope")
    assert "DAO+Gaia" in vsx_row["value"]
    assert model["vsx_depth"]["warn"] is False
