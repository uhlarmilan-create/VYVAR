"""CONSOLIDATE-01E6b: moved giant defs remain reachable through the pipeline facade."""
from __future__ import annotations

import inspect

import pipeline
import catalog_match
import frame_export
import masterstar_build
import astrometry_align


# One function per module.

def test_e6b_catalog_match_facade() -> None:
    obj = pipeline.detect_stars_and_match_catalog
    home = catalog_match.detect_stars_and_match_catalog
    assert obj is home
    assert obj.__module__ == "catalog_match"


def test_e6b_frame_export_facade() -> None:
    obj = pipeline.export_per_frame_catalogs
    home = frame_export.export_per_frame_catalogs
    assert obj is home
    assert obj.__module__ == "frame_export"


def test_e6b_masterstar_build_facade() -> None:
    obj = pipeline.generate_masterstar_and_catalog
    home = masterstar_build.generate_masterstar_and_catalog
    assert obj is home
    assert obj.__module__ == "masterstar_build"


def test_e6b_astrometry_align_facade() -> None:
    obj = pipeline._astrometry_align_impl_body
    home = astrometry_align._astrometry_align_impl_body
    assert obj is home
    assert obj.__module__ == "astrometry_align"


def test_e6b_astropipeline_stays() -> None:
    """AstroPipeline (C-C) must remain physical in pipeline.py."""
    assert pipeline.AstroPipeline.__module__ == "pipeline"


def test_e6b_fill_masterstars_follow() -> None:
    """_fill_masterstars_gaia_matched_bp_rp_from_local_db is patched on the facade
    (test_invariants_p2.py:362). The call-time follow in masterstar_build resolves
    via the facade at call time; confirm the facade binding exists and is in
    pipeline_astrometry (home module)."""
    import pipeline_astrometry
    facade = pipeline._fill_masterstars_gaia_matched_bp_rp_from_local_db
    home = pipeline_astrometry._fill_masterstars_gaia_matched_bp_rp_from_local_db
    assert facade is home
    assert facade.__module__ == "pipeline_astrometry"


def test_e6b_pipeline_py_line_count() -> None:
    """pipeline.py should be substantially reduced after E6b."""
    from pathlib import Path
    import re
    src = Path(pipeline.__file__).read_text(encoding="utf-8")
    n = len(src.splitlines())
    # Must be well under 2000 (was 7533 at E6a close; 4 giants removed)
    assert n < 2000, f"pipeline.py is {n} lines after E6b; expected < 2000"
