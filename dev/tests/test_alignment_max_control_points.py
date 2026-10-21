"""G1-F001/F002: alignment_max_control_points decoupled from detection ladder max_stars."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import AppConfig
from vyvar_alignment_frame import _alignment_as_alignment_points, _alignment_run_astroalign_points


def test_default_alignment_max_control_points_is_80():
    cfg = AppConfig()
    assert cfg.alignment_max_control_points == 80


def test_alignment_max_control_points_clamped_in_post_init(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"alignment_max_control_points": 9999}),
        encoding="utf-8",
    )
    cfg = AppConfig(project_root=tmp_path)
    assert cfg.alignment_max_control_points == 500

    (tmp_path / "config.json").write_text(
        json.dumps({"alignment_max_control_points": 3}),
        encoding="utf-8",
    )
    cfg2 = AppConfig(project_root=tmp_path)
    assert cfg2.alignment_max_control_points == 12


def test_alignment_max_control_points_from_config_json(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"alignment_max_control_points": 96}),
        encoding="utf-8",
    )
    cfg = AppConfig(project_root=tmp_path)
    assert cfg.alignment_max_control_points == 96
    d = cfg.to_dict()
    assert d["alignment_max_control_points"] == 96


def test_mcp_uses_max_control_points_not_detection_max_st():
    """300 detections + cfg cap 80 ? mcp == 80 (old ladder used min(max_st, n_fit) == 300)."""
    max_st = 300
    n_fit = 300
    max_control_points = 80
    mcp = max(12, min(int(max_control_points), n_fit))
    assert mcp == 80
    legacy_mcp = max(12, min(max_st, n_fit))
    assert legacy_mcp == 300


def test_alignment_run_astroalign_points_respects_control_point_cap():
    rng = np.random.default_rng(42)
    n = 120
    src = rng.uniform(80, 420, (n, 2)).astype(np.float32)
    tgt = src + rng.normal(0, 0.25, (n, 2)).astype(np.float32)
    img = rng.random((256, 256), dtype=np.float32)

    captured: list[int] = []
    import astroalign

    real_find = astroalign.find_transform

    def _spy_find_transform(*args, **kwargs):
        captured.append(int(kwargs.get("max_control_points", -1)))
        return real_find(*args, **kwargs)

    import unittest.mock as mock

    with mock.patch.object(astroalign, "find_transform", _spy_find_transform):
        out, err, _ = _alignment_run_astroalign_points(
            source_pts=src,
            target_pts=tgt,
            image_source=img,
            image_target=img,
            max_control_points=80,
        )
    assert err is None and out is not None
    assert captured == [80]


def test_as_alignment_points_then_mcp_with_dense_detection():
    rng = np.random.default_rng(7)
    xy = rng.uniform(50, 450, (300, 2)).astype(np.float32)
    ref = rng.uniform(50, 450, (300, 2)).astype(np.float32)
    xy_fit = _alignment_as_alignment_points(xy, label="source", log_sink=None)
    ref_fit = _alignment_as_alignment_points(ref, label="target", log_sink=None)
    n_fit = int(min(len(xy_fit), len(ref_fit)))
    mcp = max(12, min(80, n_fit))
    assert n_fit == 300
    assert mcp == 80
