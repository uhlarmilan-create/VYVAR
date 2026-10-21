"""Sparse comp fallback removed (COMP-RULES-2026-10): config keys."""

from __future__ import annotations

import json
from pathlib import Path

import config
from config import AppConfig


def test_sparse_fallback_keys_removed(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "comp_iterative_clip_enabled": True,
                "comp_sparse_fallback_enabled": True,
                "comp_sparse_fallback_min": 5,
            }
        ),
        encoding="utf-8",
    )
    cfg = AppConfig(project_root=tmp_path)
    assert not hasattr(cfg, "comp_iterative_clip_enabled")
    assert not hasattr(cfg, "comp_sparse_fallback_enabled")
    assert not hasattr(cfg, "comp_sparse_fallback_min")


def test_sparse_fallback_resolvers_removed() -> None:
    assert not hasattr(config, "resolve_comp_sparse_fallback_enabled")
    assert not hasattr(config, "resolve_comp_sparse_fallback_min")
