"""CONSOLIDATE-01E6b: SAT_LIMIT twin guard.

pipeline_catalog.py defines a numeric twin SAT_LIMIT_NO_KNEE_FRAC = 0.80
for default-arg evaluation without a module-level pipeline import.
This test asserts equality so silent divergence is caught immediately.

Rule: any numeric twin created for default-arg or import-order reasons
MUST ship with an equality-guard test in the same commit.

Permanent fix (canonical constants in a leaf module) is E-final material.
"""
from __future__ import annotations

import pipeline
import pipeline_catalog


def test_sat_limit_no_knee_frac_twin() -> None:
    assert pipeline.SAT_LIMIT_NO_KNEE_FRAC == pipeline_catalog.SAT_LIMIT_NO_KNEE_FRAC, (
        f"SAT_LIMIT_NO_KNEE_FRAC diverged: pipeline={pipeline.SAT_LIMIT_NO_KNEE_FRAC} "
        f"pipeline_catalog={pipeline_catalog.SAT_LIMIT_NO_KNEE_FRAC}"
    )
