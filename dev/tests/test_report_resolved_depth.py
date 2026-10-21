"""Report resolved-facts block: VSX scope row and the one frame depth (FRAME-DEPTH-2026-10)."""

from __future__ import annotations

from photometry_report import resolved_facts_model


def test_resolved_facts_tolerates_missing_depth() -> None:
    model = resolved_facts_model(
        {"resolved_facts": {"filter": "V", "exptime_s": 60.0}},
    )
    labels = [r["label"] for r in model["rows"]]
    assert "VSX auto-target scope" in labels
    assert "Frame faint limit (Gaia G)" not in labels
    vsx_row = next(r for r in model["rows"] if r["label"] == "VSX auto-target scope")
    assert "detection-limited" in vsx_row["value"]
    assert model["vsx_depth"]["warn"] is False


def test_resolved_facts_frame_depth_and_topup_rows() -> None:
    meta = {
        "resolved_facts": {"filter": "V"},
        "frame_depth": {
            "g_lim_frame": 19.1, "g_lim": 17.5, "g50": 17.9, "sigma_mag": 0.4,
            "method": "detector_estimate", "db_depth_g": 17.5, "limited_by_catalogue": True,
        },
        "gaia_topup": {"status": "offline", "error": "gaierror", "depth_before": 17.5},
    }
    model = resolved_facts_model(meta)
    rows = {r["label"]: r for r in model["rows"]}
    assert rows["Frame faint limit (Gaia G)"]["value"] == "19.1"
    assert "LIMITED BY THE CATALOGUE" in rows["Gaia matching depth (G)"]["source"]
    assert "NOT POSSIBLE" in rows["Gaia DB top-up (field)"]["value"]
    meta["gaia_topup"] = {"status": "ok", "g_to": 19.1, "n_new": 4321}
    rows = {r["label"]: r for r in resolved_facts_model(meta)["rows"]}
    assert "4321 new stars" in rows["Gaia DB top-up (field)"]["value"]
    meta["gaia_topup"] = {"status": "not_needed"}
    rows = {r["label"]: r for r in resolved_facts_model(meta)["rows"]}
    assert "Gaia DB top-up (field)" not in rows
