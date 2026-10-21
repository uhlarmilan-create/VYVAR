"""Report builders for the honest full-config section (PARAM-OWNERSHIP-WAVE-A STEP 4).

Pure, Streamlit/reportlab-free tests for:
  * full_config_snapshot_model -- the complete as-run snapshot grouped by phase.
  * resolved_facts_model       -- the run-effective resolved facts block.
Both must handle synthetic provenance and legacy/missing-field drafts without crashing.
"""
from __future__ import annotations

import params_registry as pr
from photometry_report import full_config_snapshot_model, resolved_facts_model


def _synthetic_meta() -> dict:
    return {
        "provenance": {
            "config_snapshot": {
                "gain": 2.0,
                "aperture_fwhm_factor": 1.4,
                "some_unknown_legacy_key": 123,
            }
        },
        "resolved_facts": {
            "site": {
                "location_id": 2, "name": "Jirny", "lat": 50.1, "lon": 14.7,
                "alt_m": 300.0, "source": "draft", "ok": True,
            },
            "gain": {"value": 2.0, "source": "header", "key": "EGAIN"},
            "read_noise": {"value": 8.5, "source": "db", "key": None},
            "saturation_adu": 60000.0,
            "plate_scale_arcsec_per_px": 1.31,
            "frame_width_px": 4096,
            "frame_height_px": 4096,
            "binning": "2x2",
            "filter": "V",
            "exptime_s": 60.0,
        },
    }


# --------------------------------------------------------------------------- #
# full_config_snapshot_model                                                   #
# --------------------------------------------------------------------------- #
def test_full_snapshot_from_run_groups_by_phase() -> None:
    model = full_config_snapshot_model(_synthetic_meta())
    assert model["fallback"] is False
    assert model["source_label"] == "run snapshot"
    assert model["n_keys"] == 3

    flat = [k for ph in model["phases"] for (k, _v) in model["by_phase"][ph]]
    assert sorted(flat) == ["aperture_fwhm_factor", "gain", "some_unknown_legacy_key"]

    # unknown/legacy key lands in 'other'; real keys land in a valid registry phase.
    assert "some_unknown_legacy_key" in [k for (k, _v) in model["by_phase"]["other"]]
    for ph in model["phases"]:
        if ph == "other":
            continue
        assert ph in pr.PHASES


def test_full_snapshot_missing_snapshot_falls_back_to_live() -> None:
    model = full_config_snapshot_model({})
    assert model["fallback"] is True
    assert model["source_label"] == "live (no run snapshot)"
    # live AppConfig has many keys; grouping must still be well-formed
    assert model["n_keys"] > 0
    assert model["phases"]


def test_full_snapshot_none_meta_does_not_crash() -> None:
    model = full_config_snapshot_model(None)
    assert model["fallback"] is True
    assert isinstance(model["by_phase"], dict)


def test_full_snapshot_completeness_note_trigger() -> None:
    reg = pr.load_registry()
    # Complete snapshot (every registry key present) -> no omitted keys, no note.
    complete = {k: 0 for k in reg}
    m = full_config_snapshot_model({"provenance": {"config_snapshot": complete}})
    assert m["registry_count"] == len(reg)
    assert m["omitted_keys"] == []
    # Incomplete snapshot -> omitted_keys names exactly the missing registry keys.
    partial = dict(list(complete.items())[:-3])
    m2 = full_config_snapshot_model({"provenance": {"config_snapshot": partial}})
    assert m2["omitted_keys"] == sorted(set(reg) - set(partial))
    assert len(m2["omitted_keys"]) == 3


def test_complete_config_snapshot_covers_all_public_fields() -> None:
    import dataclasses
    import json

    import photometry_core as pc
    from config import AppConfig

    cfg = AppConfig()
    snap = pc._complete_config_snapshot(cfg, cfg.to_dict())

    public = {f.name for f in dataclasses.fields(cfg) if not f.name.startswith("_")}
    assert public <= set(snap), f"missing public fields: {sorted(public - set(snap))}"
    assert set(pr.load_registry()) <= set(snap), "snapshot must cover every registry key"
    # backfilled derived fields are present and JSON-safe (Path -> str)
    assert "project_root" in snap and isinstance(snap["project_root"], str)
    assert "qc_preprocess_workers" in snap
    assert "plate_solve_fov_deg" in snap
    json.dumps(snap)  # must be JSON-serializable (would raise on a stray Path)


# --------------------------------------------------------------------------- #
# resolved_facts_model                                                         #
# --------------------------------------------------------------------------- #
def _row(model: dict, label: str) -> dict:
    return next(r for r in model["rows"] if r["label"] == label)


def test_resolved_facts_from_provenance() -> None:
    model = resolved_facts_model(_synthetic_meta())
    assert model["fallback"] is False
    assert not model["warnings"]

    site = _row(model, "Site (LOCATION)")
    assert "Jirny" in site["value"] and "id=2" in site["value"]
    assert site["source"] == "draft"

    gain = _row(model, "Gain (e-/ADU)")
    assert gain["value"] == "2"
    assert "header" in gain["source"] and "[EGAIN]" in gain["source"]

    assert _row(model, "Read noise (e-)")["value"] == "8.5"
    assert _row(model, "Saturation (ADU)")["value"] == "60000"
    assert _row(model, "Plate scale (arcsec/px)")["value"] == "1.31"
    assert _row(model, "Frame (px)")["value"] == "4096 x 4096"
    assert _row(model, "Binning")["value"] == "2x2"
    assert _row(model, "Filter")["value"] == "V"
    assert _row(model, "Exposure (s)")["value"] == "60"


def test_resolved_facts_legacy_fallback_to_dynamic_params() -> None:
    # No resolved_facts block: recover gain/RN/plate from dynamic_params, site from observer_location.
    meta = {
        "observer_location": {"location_id": 1, "name": "Dablice", "lat": 50.2, "lon": 14.5, "alt_m": 280.0, "source": "config"},
        "dynamic_params": {"gain": 1.5, "read_noise": 9.0, "plate_scale_arcsec_px": 1.29},
    }
    model = resolved_facts_model(meta)
    assert model["fallback"] is True
    assert model["warnings"]
    assert _row(model, "Gain (e-/ADU)")["value"] == "1.5"
    assert _row(model, "Read noise (e-)")["value"] == "9"
    assert _row(model, "Plate scale (arcsec/px)")["value"] == "1.29"
    site = _row(model, "Site (LOCATION)")
    assert "Dablice" in site["value"]
    # missing fields render as '-' and never crash
    assert _row(model, "Binning")["value"] == "-"
    assert _row(model, "Filter")["value"] == "-"


def test_resolved_facts_empty_meta_does_not_crash() -> None:
    model = resolved_facts_model(None)
    assert model["fallback"] is True
    # 9 core facts + AUTO-VSX-LIMIT depth comparison row
    assert len(model["rows"]) == 10
    assert all(set(r.keys()) == {"label", "value", "source"} for r in model["rows"])
    assert any(r["label"] == "VSX auto-target scope" for r in model["rows"])


# --------------------------------------------------------------------------- #
# provenance writer: _build_phase2a_resolved_facts (metadata capture)          #
# --------------------------------------------------------------------------- #
class _FakeResolved:
    def __init__(self, value, source, key, ok):  # noqa: ANN001
        self.value, self.source, self.key, self.ok = value, source, key, ok


class _FakeSite:
    def __init__(self, lat, lon, elev, source, ok):  # noqa: ANN001
        self.lat, self.lon, self.elev, self.source, self.ok = lat, lon, elev, source, ok


class _FakeCfg:
    observer_lat = 49.0
    observer_lon = 15.0
    observer_alt_m = 250.0
    observer_location_id = 2
    observer_location_name = "Jirny"


def test_build_resolved_facts_writer_captures_sources() -> None:
    import photometry_core as pc

    facts = pc._build_phase2a_resolved_facts(
        cfg=_FakeCfg(),
        gain_res=_FakeResolved(2.0, "header", "EGAIN", True),
        rn_res=_FakeResolved(8.5, "db", None, True),
        gain_value=2.0,
        rn_value=8.5,
        site=_FakeSite(50.1, 14.7, 300.0, "draft", True),
        sat_limit=60000.0,
        plate_scale_arcsec=1.31,
        frame_width_px=4096,
        frame_height_px=4096,
        ms_header={"FILTER": "V", "EXPTIME": 60.0, "XBINNING": 2, "YBINNING": 2},
        obs_group="NoFilter_60_2",
    )
    assert facts["gain"] == {"value": 2.0, "source": "header", "key": "EGAIN"}
    assert facts["read_noise"] == {"value": 8.5, "source": "db", "key": None}
    assert facts["site"]["source"] == "draft" and facts["site"]["name"] == "Jirny"
    assert facts["site"]["lat"] == 50.1  # from resolver, not cfg
    assert facts["saturation_adu"] == 60000.0
    assert facts["frame_width_px"] == 4096 and facts["frame_height_px"] == 4096
    assert facts["binning"] == "2x2"
    assert facts["filter"] == "V"
    assert facts["exptime_s"] == 60.0


def test_build_resolved_facts_writer_fallbacks_when_unresolved() -> None:
    import photometry_core as pc

    facts = pc._build_phase2a_resolved_facts(
        cfg=_FakeCfg(),
        gain_res=_FakeResolved(None, None, None, False),
        rn_res=_FakeResolved(None, None, None, False),
        gain_value=None,
        rn_value=None,
        site=_FakeSite(None, None, None, "unresolved", False),
        sat_limit=None,
        plate_scale_arcsec=None,
        frame_width_px=None,
        frame_height_px=None,
        ms_header=None,
        obs_group="V_120_1",
    )
    # gain/RN unresolved -> source "default"; site falls back to cfg coords; filter -> obs_group
    assert facts["gain"]["source"] == "default"
    assert facts["read_noise"]["source"] == "default"
    assert facts["site"]["lat"] == 49.0 and facts["site"]["ok"] is False
    assert facts["filter"] == "V_120_1"
    assert facts["binning"] is None
    assert facts["exptime_s"] is None
