"""Tests for blind index tier config and ordering."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from config import AppConfig
from vyvar_blind_series import (
    build_tiers_from_config,
    estimate_rho_img_deg2,
    order_tiers_for_image,
    target_density_deg2,
)


def test_target_density_deg2() -> None:
    assert target_density_deg2(cell_deg=1.0, stars_per_cell=95) == pytest.approx(95.0)
    assert target_density_deg2(cell_deg=2.0, stars_per_cell=16) == pytest.approx(4.0)


def test_build_tiers_from_config_no_manifest(tmp_path: Path) -> None:
    fine = tmp_path / "gaia_triangles_fine.pkl"
    wide = tmp_path / "gaia_triangles_wide.pkl"
    fine.touch()
    wide.touch()
    cfg = AppConfig(project_root=tmp_path)
    cfg.blind_index_fine_path = str(fine)
    cfg.blind_index_wide_path = str(wide)
    tiers = build_tiers_from_config(cfg)
    assert len(tiers) == 2
    assert tiers[0]["name"] == "fine"
    assert tiers[1]["name"] == "wide"
    assert tiers[0]["target_density_deg2"] == pytest.approx(95.0)
    assert tiers[1]["target_density_deg2"] == pytest.approx(4.0)


def test_ordering_does_not_load_pkl(tmp_path: Path) -> None:
    fine = tmp_path / "fine.pkl"
    wide = tmp_path / "wide.pkl"
    fine.touch()
    wide.touch()
    cfg = AppConfig(project_root=tmp_path)
    cfg.blind_index_fine_path = str(fine)
    cfg.blind_index_wide_path = str(wide)
    tiers = build_tiers_from_config(cfg)

    def _boom(*_a, **_k):
        raise AssertionError("pickle.load must not run for tier ordering")

    with patch("vyvar_blind_series.pickle.load", side_effect=_boom):
        wide_first = order_tiers_for_image(tiers, rho_img_deg2=4.0, plate_scale_arcsec_per_px=9.77)
        fine_first = order_tiers_for_image(tiers, rho_img_deg2=95.0, plate_scale_arcsec_per_px=1.3)
    assert wide_first[0]["name"] == "wide"
    assert fine_first[0]["name"] == "fine"


def test_config_migration_from_legacy_blind_index_path(tmp_path: Path) -> None:
    gaia = tmp_path / "GAIA_DR3"
    gaia.mkdir(parents=True)
    legacy = gaia / "gaia_triangles.pkl"
    legacy.touch()
    (gaia / "gaia_triangles_fine.pkl").touch()
    (gaia / "gaia_triangles_wide.pkl").touch()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "blind_index_path": str(legacy),
                "blind_index_series": str(gaia / "blind_index_series.json"),
            }
        ),
        encoding="utf-8",
    )
    cfg = AppConfig(project_root=tmp_path)
    assert cfg.blind_index_fine_path.endswith("gaia_triangles_fine.pkl")
    assert cfg.blind_index_wide_path.endswith("gaia_triangles_wide.pkl")
    assert cfg.blind_index_path == cfg.blind_index_fine_path


def test_estimate_rho_img_positive() -> None:
    rho = estimate_rho_img_deg2(plate_scale_arcsec_per_px=1.3, fov_deg=1.1, img_budget=80)
    assert rho > 10.0
