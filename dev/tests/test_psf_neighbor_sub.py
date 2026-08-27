"""Tests for psf_neighbor_sub joint-fit core."""
from __future__ import annotations

import numpy as np

from config import AppConfig
from psf_neighbor_sub import neighbor_sub_target_flux
from tests.validation.a9_core import (
    A9_CONTEXTS,
    A9Context,
    STAMP_C,
    build_blend_frame,
    measure_cell,
    score_neighbor_sub_cell,
)
from tests.validation.score import RNG_SEEDS


def _enabled_cfg() -> AppConfig:
    c = AppConfig()
    c.psf_neighbor_sub_enabled = True
    return c


def _a9_envelope_ctx(name: str) -> A9Context:
    """A9 joint-fit envelope at f=1.9 / annulus 4.75/9, not production APERTURE-01d radii."""
    base = A9_CONTEXTS[name]
    return A9Context(
        name=base.name,
        fwhm_px=base.fwhm_px,
        plate_scale_arcsec=base.plate_scale_arcsec,
        aperture_fwhm_factor=1.9,
        annulus_inner_fwhm=4.75,
        annulus_outer_fwhm=9.0,
    )


def test_joint_recovers_high_value_ideal():
    ctx = _a9_envelope_ctx("coarse")
    rng = np.random.default_rng(RNG_SEEDS["gen_a9"])
    tflux = ctx.target_flux_adu()
    nflux = tflux * 10.0 ** 0.8
    r_ap, r_in, r_out = ctx.radii_px()
    biases = []
    for _ in range(10):
        frame = build_blend_frame(
            ctx, target_flux=tflux, neighbour_flux=nflux, sep_fwhm=1.0, rng=rng
        )
        res = neighbor_sub_target_flux(
            frame,
            target_xy=(float(STAMP_C), float(STAMP_C)),
            neighbour_xys=[(float(STAMP_C), float(STAMP_C + ctx.fwhm_px))],
            fwhm_px=ctx.fwhm_px,
            r_ap=r_ap,
            r_in=r_in,
            r_out=r_out,
            delta_mag_nn=-2.0,
            nn_dist_fwhm=1.0,
            target_mag=13.0,
            nn_mag=11.0,
            flux_zp=25.0,
            cfg=_enabled_cfg(),
        )
        assert not res.refused
        biases.append((res.target_flux / tflux - 1.0) * 100.0)
    assert abs(float(np.median(biases))) < 12.0


def test_bright_close_regime_refuses_sep10_dm3():
    """Preemptive refuse for brightest neighbour at tightest HV separation (367 edge cell)."""
    ctx = A9_CONTEXTS["draft367"]
    rng = np.random.default_rng(RNG_SEEDS["gen_a9"] + 5)
    tflux = ctx.target_flux_adu()
    nflux = tflux * 10.0 ** 1.2
    r_ap, r_in, r_out = ctx.radii_px()
    frame = build_blend_frame(
        ctx,
        target_flux=tflux,
        neighbour_flux=nflux,
        sep_fwhm=1.0,
        rng=rng,
        inject_beta=2.5,
        star_fwhm_scale=5.396 / 6.0203,
        neighbour_fwhm_scale=5.396 / 6.0203,
    )
    res = neighbor_sub_target_flux(
        frame,
        target_xy=(float(STAMP_C), float(STAMP_C)),
        neighbour_xys=[(float(STAMP_C), float(STAMP_C + ctx.fwhm_px))],
        fwhm_px=ctx.fwhm_px * (5.3925 / 6.0203),
        r_ap=r_ap,
        r_in=r_in,
        r_out=r_out,
        delta_mag_nn=-3.0,
        nn_dist_fwhm=1.0,
        target_mag=13.0,
        nn_mag=10.0,
        flux_zp=25.0,
        cfg=_enabled_cfg(),
    )
    assert res.refused
    assert res.refuse_reason == "bright_close_regime"
    assert not res.neighbor_subtracted
    assert res.target_flux == res.plain_target_flux


def test_bright_close_regime_allows_sep13_dm3():
    ctx = A9_CONTEXTS["draft367"]
    from tests.validation.a9_core import A9Cell

    cell = A9Cell(sep_fwhm=1.3, delta_mag=-3, context="draft367")
    sub = measure_cell(cell, ctx, mode="neighbor_sub", psf_variant="draft367", n_frames=8)
    assert not sub.neighbor_sub_refused or sub.refuse_reason != "bright_close_regime"


def test_refuse_zone_sep_08_inclusive():
    ctx = _a9_envelope_ctx("coarse")
    rng = np.random.default_rng(RNG_SEEDS["gen_a9"] + 2)
    tflux = ctx.target_flux_adu()
    r_ap, r_in, r_out = ctx.radii_px()
    frame = build_blend_frame(
        ctx, target_flux=tflux, neighbour_flux=tflux * 10, sep_fwhm=0.8, rng=rng
    )
    res = neighbor_sub_target_flux(
        frame,
        target_xy=(float(STAMP_C), float(STAMP_C)),
        neighbour_xys=[(float(STAMP_C), float(STAMP_C + 0.8 * ctx.fwhm_px))],
        fwhm_px=ctx.fwhm_px,
        r_ap=r_ap,
        r_in=r_in,
        r_out=r_out,
        delta_mag_nn=-2.5,
        nn_dist_fwhm=0.8,
        target_mag=13.0,
        nn_mag=10.5,
        flux_zp=25.0,
        cfg=_enabled_cfg(),
    )
    assert res.refused
    assert res.refuse_reason == "sep_floor"


def test_refuse_zone_sep_05():
    ctx = _a9_envelope_ctx("coarse")
    rng = np.random.default_rng(RNG_SEEDS["gen_a9"] + 1)
    tflux = ctx.target_flux_adu()
    r_ap, r_in, r_out = ctx.radii_px()
    frame = build_blend_frame(
        ctx, target_flux=tflux, neighbour_flux=tflux * 100, sep_fwhm=0.5, rng=rng
    )
    res = neighbor_sub_target_flux(
        frame,
        target_xy=(float(STAMP_C), float(STAMP_C)),
        neighbour_xys=[(float(STAMP_C), float(STAMP_C + 0.5 * ctx.fwhm_px))],
        fwhm_px=ctx.fwhm_px,
        r_ap=r_ap,
        r_in=r_in,
        r_out=r_out,
        delta_mag_nn=-3.0,
        nn_dist_fwhm=0.5,
        cfg=_enabled_cfg(),
    )
    assert res.refused


def test_a9_ideal_scores_high_value_cell():
    ctx = _a9_envelope_ctx("coarse")
    from tests.validation.a9_core import A9Cell

    cell = A9Cell(sep_fwhm=1.3, delta_mag=-2, context="coarse")
    sub = measure_cell(cell, ctx, mode="neighbor_sub", psf_variant="ideal", n_frames=10)
    assert sub.pass_future_neighbor_sub is True


def test_mismatch_degrades_vs_ideal():
    ctx = _a9_envelope_ctx("coarse")
    from tests.validation.a9_core import A9Cell

    cell = A9Cell(sep_fwhm=1.3, delta_mag=-3, context="coarse")
    ideal = measure_cell(cell, ctx, mode="neighbor_sub", psf_variant="ideal", n_frames=10)
    mismatch = measure_cell(cell, ctx, mode="neighbor_sub", psf_variant="mismatch", n_frames=10)
    assert abs(ideal.contamination_excess_pct) <= abs(mismatch.contamination_excess_pct) + 5.0


def test_refuse_neighbor_overfit_realistic_mismatch():
    """Realistic PSF mismatch can over-estimate neighbour; catalog anchor must refuse."""
    ctx = _a9_envelope_ctx("coarse")
    from tests.validation.a9_core import psf_variant_spec

    pspec = psf_variant_spec("realistic")
    rng = np.random.default_rng(RNG_SEEDS["gen_a9"] + 3)
    tflux = ctx.target_flux_adu()
    nflux = tflux * 10.0 ** 0.8
    r_ap, r_in, r_out = ctx.radii_px()
    refused = 0
    for _ in range(8):
        frame = build_blend_frame(
            ctx,
            target_flux=tflux,
            neighbour_flux=nflux,
            sep_fwhm=1.0,
            rng=rng,
            inject_beta=pspec.inject_beta,
            star_fwhm_scale=pspec.star_fwhm_scale,
            inject_ellip=pspec.inject_ellip,
            inject_theta=pspec.inject_theta,
        )
        res = neighbor_sub_target_flux(
            frame,
            target_xy=(float(STAMP_C), float(STAMP_C)),
            neighbour_xys=[(float(STAMP_C), float(STAMP_C + ctx.fwhm_px))],
            fwhm_px=ctx.fwhm_px,
            r_ap=r_ap,
            r_in=r_in,
            r_out=r_out,
            delta_mag_nn=-2.0,
            nn_dist_fwhm=1.0,
            target_mag=13.0,
            nn_mag=11.0,
            flux_zp=25.0,
            cfg=_enabled_cfg(),
        )
        if res.refused:
            refused += 1
            assert res.refuse_reason in (
                "neighbor_overfit",
                "target_undershoot",
                "no_improvement",
            )
            assert not res.neighbor_subtracted
    assert refused >= 6


def test_refuse_target_undershoot_when_cleaned_too_faint():
    ctx = _a9_envelope_ctx("coarse")
    rng = np.random.default_rng(RNG_SEEDS["gen_a9"] + 4)
    tflux = ctx.target_flux_adu()
    nflux = tflux * 100.0
    r_ap, r_in, r_out = ctx.radii_px()
    frame = build_blend_frame(
        ctx, target_flux=tflux, neighbour_flux=nflux, sep_fwhm=1.5, rng=rng
    )
    res = neighbor_sub_target_flux(
        frame,
        target_xy=(float(STAMP_C), float(STAMP_C)),
        neighbour_xys=[(float(STAMP_C), float(STAMP_C + 1.5 * ctx.fwhm_px))],
        fwhm_px=ctx.fwhm_px,
        r_ap=r_ap,
        r_in=r_in,
        r_out=r_out,
        delta_mag_nn=-5.0,
        nn_dist_fwhm=1.5,
        target_mag=13.0,
        nn_mag=8.0,
        flux_zp=25.0,
        cfg=_enabled_cfg(),
    )
    if not res.neighbor_subtracted:
        assert res.refused
        assert res.refuse_reason in (
            "target_undershoot",
            "neighbor_overfit",
            "no_improvement",
            "bright_close_regime",
        )


def test_score_refuse_zone():
    assert score_neighbor_sub_cell("REFUSE", 500.0, 0.0, refused=True, neighbor_subtracted=False, criterion={})
    assert not score_neighbor_sub_cell("REFUSE", 500.0, 50.0, refused=False, neighbor_subtracted=True, criterion={})
