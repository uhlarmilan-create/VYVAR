"""Unit tests for sparse-only comp fallback config + wiring."""

from __future__ import annotations

from config import AppConfig, resolve_comp_sparse_fallback_enabled, resolve_comp_sparse_fallback_min


def test_sparse_fallback_legacy_alias() -> None:
    cfg = AppConfig()
    cfg.comp_sparse_fallback_enabled = False
    cfg.comp_iterative_clip_enabled = True
    assert resolve_comp_sparse_fallback_enabled(cfg) is True


def test_sparse_fallback_min_defaults_to_n_comp_min() -> None:
    cfg = AppConfig()
    cfg.comp_sparse_fallback_min = 0
    assert resolve_comp_sparse_fallback_min(cfg, n_comp_min=3, n_comp_max=7) == 3


def test_sparse_fallback_min_clamped() -> None:
    cfg = AppConfig()
    cfg.comp_sparse_fallback_min = 99
    assert resolve_comp_sparse_fallback_min(cfg, n_comp_min=3, n_comp_max=7) == 7
