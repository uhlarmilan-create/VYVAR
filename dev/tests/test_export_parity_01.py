# -*- coding: ascii -*-
"""NightRunParams defaults preserve W2 behaviour (EXPORT-PARITY-01 v2 3a)."""

from __future__ import annotations

from pathlib import Path

from night_run import NightRunParams


def test_nightrunparams_new_fields_default_preserve_w2() -> None:
    p = NightRunParams(source_dir=Path("."), equipment_id=1, telescope_id=1)
    assert p.optics is None
    assert p.location_source_hint is None
    assert p.masterdark_validity_days is None
    assert p.masterflat_validity_days is None
    assert p.apply_smart_plan_flat_fallbacks is False
    assert p.flat_fallback_choices is None
    assert p.roundness_reject_above == 1.25
    assert p.post_platesolve_hook is None
