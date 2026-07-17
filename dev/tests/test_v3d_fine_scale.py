"""V3d fine-scale PSF-vs-aperture-vs-truth harness tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.validation.v3d_fine_scale import (
    V3dFineConfig,
    aperture_correction_factor,
    mag_to_flux,
    run_v3d_bias_decomposition,
    run_v3d_fine_scale,
    write_bias_decomposition_report,
    write_v3d_report,
)


@pytest.fixture
def fast_cfg() -> V3dFineConfig:
    return V3dFineConfig(n_real=6, mags=(12, 14, 16))


def test_mag_to_flux():
    assert mag_to_flux(13.0) > mag_to_flux(14.0)


def test_aperture_correction_positive(fast_cfg):
    assert aperture_correction_factor(fast_cfg) > 1.0


def test_v3d_run_structure(fast_cfg, tmp_path):
    result = run_v3d_fine_scale(fast_cfg, work_dir=tmp_path / "work")
    assert result["status"] in ("PASS", "FLAG", "FAIL")
    assert len(result["mag_stats"]) == 3
    assert result["self_check"]["pass"] is True
    assert 0.85 <= result["mismatch_ratio"] <= 1.15
    for s in result["mag_stats"]:
        assert "psf_bias_pct" in s
        assert "psf_scatter_pct" in s
        assert s["precision_winner"] in ("PSF", "APER")


def test_write_v3d_report(fast_cfg, tmp_path):
    result = run_v3d_fine_scale(fast_cfg, work_dir=tmp_path / "work")
    jp, mp = write_v3d_report(tmp_path, result)
    assert jp.is_file() and mp.is_file()
    text = mp.read_text(encoding="ascii")
    assert "Pillar 1" in text
    assert "Pillar 3" in text


def test_v3d_pillar3_calibration_present(fast_cfg, tmp_path):
    result = run_v3d_fine_scale(fast_cfg, work_dir=tmp_path / "work")
    calibrated = [s for s in result["mag_stats"] if s.get("psf_cal_ratio") is not None]
    assert len(calibrated) >= 2


def test_bias_decomposition_structure(fast_cfg, tmp_path):
    result = run_v3d_bias_decomposition(fast_cfg, work_dir=tmp_path / "bias_work")
    assert len(result["pre_ac_stats"]) == 3
    assert len(result["post_ac_stats"]) == 3
    assert result["diagnosis"]["psf_ac_mag_dependent"] is False
    ac = float(result["psf_aperture_correction_factor"])
    assert 0.5 < ac < 1.5
    bright_post = [s for s in result["post_ac_stats"] if s["mag"] <= 13]
    assert bright_post and max(abs(s["bias_pct"]) for s in bright_post) < 5.0
    assert result["diagnosis"]["localized_cause"] in (
        "fit_stage_pre_ac",
        "fit_background_border_median",
        "aperture_correction_stage",
        "undetermined",
    )


def test_psf_sky_method_residual_annulus_on_v3d_frame(fast_cfg, tmp_path):
    from astropy.io import fits

    from tests.validation.v3d_fine_scale import (
        build_epsf_training_frame,
        build_isolated_frame,
        mag_to_flux,
        write_epsf_artifacts,
    )
    from psf_photometry import psf_photometry_stars
    import pandas as pd

    rng = np.random.default_rng(fast_cfg.rng_seed)
    epsf_frame, epsf_cat = build_epsf_training_frame(rng, fast_cfg)
    epsf_path = write_epsf_artifacts(tmp_path / "epsf", epsf_frame, epsf_cat, fast_cfg)
    frame = build_isolated_frame(mag_to_flux(14, fast_cfg.zp), rng, fast_cfg)
    pos = pd.DataFrame(
        [{"catalog_id": "inj", "name": "t", "x": float(fast_cfg.stamp_c), "y": float(fast_cfg.stamp_c)}]
    )
    df = psf_photometry_stars(
        frame,
        fits.Header(),
        pos,
        epsf_path,
        cutout_size=fast_cfg.cutout_size(),
        apply_aperture_correction=False,
        grouper_enabled=False,
        quality_fallback_enabled=False,
    )
    assert df.iloc[0].get("psf_sky_method") == "residual_annulus"
    assert df.iloc[0].get("psf_weight_mode") == "sky_only"
    assert df.iloc[0].get("psf_err_mode") == "sandwich_skyonly"


def test_bias_decomposition_v2_structure(fast_cfg, tmp_path):
    from tests.validation.v3d_bias_decomposition_v2 import (
        run_v3d_bias_decomposition_v2,
        write_v3d_bias_decomposition_v2_report,
    )

    result = run_v3d_bias_decomposition_v2(fast_cfg, work_dir=tmp_path / "v2_work")
    assert result["t1"]["branch"] in ("T2_T3_deterministic", "T4_noise_driven")
    assert "identified_cause" in result["decision"]
    if result["t1"]["branch"] == "T2_T3_deterministic":
        assert result["t2"] is not None
        assert result["t3"] is not None
    mp = write_v3d_bias_decomposition_v2_report(tmp_path, result)
    assert mp.is_file()
    assert "T1 -- Noiseless" in mp.read_text(encoding="ascii")


def test_write_bias_decomposition_report(fast_cfg, tmp_path):
    result = run_v3d_bias_decomposition(fast_cfg, work_dir=tmp_path / "bias_work")
    jp, mp = write_bias_decomposition_report(tmp_path, result)
    assert jp.is_file() and mp.is_file()
    text = mp.read_text(encoding="ascii")
    assert "pre-AC vs post-AC" in text
    assert "Background sensitivity" in text
