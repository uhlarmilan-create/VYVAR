"""Tests for blind index series manifest and tier ordering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vyvar_blind_series import (
    estimate_rho_img_deg2,
    load_series_manifest,
    order_tiers_for_image,
    target_density_deg2,
)


def test_target_density_deg2() -> None:
    assert target_density_deg2(cell_deg=1.0, stars_per_cell=95) == pytest.approx(95.0)
    assert target_density_deg2(cell_deg=2.0, stars_per_cell=16) == pytest.approx(4.0)


def test_order_tiers_prefers_matching_density(tmp_path: Path) -> None:
    manifest = {
        "tiers": [
            {"name": "fine", "path": "fine.pkl", "target_density_deg2": 95, "cell_deg": 1, "stars_per_cell": 95},
            {"name": "wide", "path": "wide.pkl", "target_density_deg2": 4, "cell_deg": 2, "stars_per_cell": 16},
        ]
    }
    p = tmp_path / "series.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    tiers = load_series_manifest(p)
    wide_first = order_tiers_for_image(tiers, rho_img_deg2=4.0, plate_scale_arcsec_per_px=9.77)
    assert wide_first[0]["name"] == "wide"
    fine_first = order_tiers_for_image(tiers, rho_img_deg2=95.0, plate_scale_arcsec_per_px=1.3)
    assert fine_first[0]["name"] == "fine"


def test_estimate_rho_img_positive() -> None:
    rho = estimate_rho_img_deg2(plate_scale_arcsec_per_px=1.3, fov_deg=1.1, img_budget=80)
    assert rho > 10.0
