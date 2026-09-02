"""CONSOLIDATE-01E2: moved defs remain reachable through the facade."""

from __future__ import annotations

import photometry
import photometry_core
import photometry_provenance


PHOTOMETRY_E2_PROVENANCE: tuple[str, ...] = (
    "_resolve_git_provenance",
    "_build_pipeline_provenance_block",
    "classify_git_dirty_paths",
    "_porcelain_status_by_path",
    "_is_import_relevant_py_path",
    "_complete_config_snapshot",
    "_json_safe_snapshot_value",
    "merge_photometry_pipeline_meta",
)


def test_e2_provenance_facade_getattr() -> None:
    for name in PHOTOMETRY_E2_PROVENANCE:
        obj = getattr(photometry_core, name)
        assert callable(obj), name
        assert obj.__module__ == "photometry_provenance", name


def test_e2_merge_in_star_import_not_required() -> None:
    """merge is not in __all__; facade getattr is the contract."""
    assert hasattr(photometry_core, "merge_photometry_pipeline_meta")
    assert photometry_core.merge_photometry_pipeline_meta is (
        photometry_provenance.merge_photometry_pipeline_meta
    )


def test_e2_resolve_git_follow_proxy() -> None:
    """risk_register: test_f431 patches photometry_core._resolve_git_provenance."""
    assert photometry_core._resolve_git_provenance is not None
    assert callable(photometry_provenance._resolve_git_provenance)
