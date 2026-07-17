"""Tests for the PDF Configuration-page model (PARAMS-REGISTRY-UI STEP 4).

Deviation table is built from a run provenance snapshot: only keys whose snapshot
value differs from the current dataclass default are listed; unknown/legacy snapshot
keys are collected separately without crashing; a missing snapshot triggers the
labelled live-config fallback.
"""
from __future__ import annotations

from photometry_report import config_deviation_model


def test_deviation_table_from_synthetic_snapshot() -> None:
    meta = {
        "provenance": {
            "git_hash": "deadbee",
            "git_dirty_code": False,
            "stamped_at_utc": "2026-07-17T00:00:00+00:00",
            "entry_point": "run_full_photometry_pipeline",
            "labbe_rng_seed_policy": "content_frame_hash_v1",
            "config_snapshot": {
                # two genuine deviations from dataclass defaults ...
                "masterdark_validity_days": 999,   # default 90
                "gain": 3.14,                       # default 1.0
                # ... one key left at its default (must NOT be listed) ...
                "masterflat_validity_days": 200,    # default 200
                # ... and one legacy/removed key (unknown) ...
                "aperture_px": 5.75,
                "preprocess_sky_surface_order": 2,
            },
        },
        "field_density_class": "dense",
    }
    model = config_deviation_model(meta)

    assert model["fallback"] is False
    assert model["source_label"] == "run snapshot"

    modified_keys = {r["key"] for r in model["rows"]}
    assert modified_keys == {"masterdark_validity_days", "gain"}
    assert "masterflat_validity_days" not in modified_keys

    assert model["unknown_keys"] == ["aperture_px"]
    assert model["header"]["n_modified"] == 2
    assert model["header"]["git_hash"] == "deadbee"
    assert model["header"]["git_dirty_code"] is False
    assert model["header"]["entry_point"] == "run_full_photometry_pipeline"
    assert model["fingerprint"]["preprocess_sky_surface_order"] == 2
    assert model["fingerprint"]["seed_policy"] == "content_frame_hash_v1"
    assert model["fingerprint"]["density_profile"] == "dense"

    # value/default pairing is carried through for rendering
    by_key = {r["key"]: r for r in model["rows"]}
    assert by_key["masterdark_validity_days"]["value"] == 999
    assert by_key["masterdark_validity_days"]["default"] == 90


def test_missing_snapshot_uses_labelled_live_fallback() -> None:
    model = config_deviation_model(None)
    assert model["fallback"] is True
    assert model["source_label"] == "live (no run snapshot)"
    # header provenance fields are absent (no run to source them from)
    assert model["header"]["git_hash"] is None
    assert model["header"]["entry_point"] is None
    # still returns a usable (non-crashing) structure
    assert isinstance(model["rows"], list)
    assert isinstance(model["unknown_keys"], list)


def test_empty_snapshot_dict_falls_back() -> None:
    # provenance present but config_snapshot empty -> treated as missing (fallback)
    model = config_deviation_model({"provenance": {"config_snapshot": {}}})
    assert model["fallback"] is True
    assert model["source_label"] == "live (no run snapshot)"
