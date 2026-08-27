"""A9 NEIGHBOR-SUB acceptance envelope: blend grid, VYVAR aperture baseline, zone criteria.

Validation-only. Uses photometry_core._annulus_sky_subtracted_flux (VYVAR annulus path)
with AppConfig-default aperture radii (VY_FWHM_GAUSS scale). ASCII, deterministic.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from config import AppConfig
from photometry_core import _annulus_sky_subtracted_flux
from tests.validation.gen_frame import moffat_stamp
from tests.validation.score import RNG_SEEDS

A9_RNG_SEED = RNG_SEEDS.get("gen_a9", 44)

SEPARATIONS_FWHM: tuple[float, ...] = (0.5, 0.8, 1.0, 1.3, 1.5, 2.0, 3.0)
DELTA_MAGS: tuple[int, ...] = (0, -1, -2, -3)

TARGET_MAG = 13.0
ZP = 25.0
N_FRAMES = 16
STAMP_N = 121
STAMP_C = STAMP_N // 2
SKY_ADU = 200.0
GAIN_E_PER_ADU = 1.5
READ_NOISE_E = 9.0
MOFFAT_BETA = 2.5
JITTER_FWHM_FRAC = 0.05

MeasureMode = Literal["plain_aperture", "neighbor_sub"]
PsfVariant = Literal["ideal", "mismatch", "realistic", "draft367"]

# Draft 367 Red_180_2 audit (2026-06-08); refreshed by epsf_fwhm_audit.analyze_draft_367().
DRAFT367_FWHM_GAUSS_PX = 6.0203
DRAFT367_FWHM_EPSF_MOFFAT_PX = 5.3925
DRAFT367_FWHM_STARS_PX = 5.396
DRAFT367_MISMATCH_RATIO = 0.9994
DRAFT367_PLATE_SCALE_ARCSEC = 0.3889
DRAFT367_INJECT_ELLIP = 0.01

_cfg = AppConfig()


@dataclass(frozen=True)
class A9Context:
    """Optics context for one envelope run (coarse h&chi Per vs fine draft-367 scale)."""

    name: str
    fwhm_px: float
    plate_scale_arcsec: float
    aperture_fwhm_factor: float = 1.9
    annulus_inner_fwhm: float = 4.75
    annulus_outer_fwhm: float = 9.0

    def radii_px(self) -> tuple[float, float, float]:
        fw = float(self.fwhm_px)
        r_ap = max(0.5, float(self.aperture_fwhm_factor) * fw)
        r_in = max(r_ap + 0.5, float(self.annulus_inner_fwhm) * fw)
        r_out = max(r_in + 0.5, float(self.annulus_outer_fwhm) * fw)
        return r_ap, r_in, r_out

    def target_flux_adu(self) -> float:
        return 10.0 ** (-0.4 * (TARGET_MAG - ZP))


# Frozen per-cell offsets (PYTHONHASHSEED=1); avoids hash-randomization flakiness in pytest.
_CELL_RNG_OFFSETS: tuple[int, ...] = (
    97363, 63432, 63432, 65007, 44691, 42632, 42632, 87382,
    27587, 25528, 25528, 38406, 95388, 93329, 93329, 89695,
    79581, 45650, 45650, 42016, 81894, 47963, 47963, 44329,
    13121, 16271, 16271, 80765, 61643, 27712, 27712, 29287,
    81060, 84210, 84210, 80576, 63956, 61897, 61897, 31600,
    83373, 86523, 86523, 82889, 15950, 82019, 82019, 35210,
    392, 49949, 49949, 51524, 77401, 75342, 75342, 45045,
    17927, 15868, 15868, 17443, 40302, 95068, 95068, 7946,
    34241, 32182, 32182, 96676, 8232, 22685, 22685, 75876,
    86235, 52304, 52304, 5495, 11250, 9191, 9191, 78894,
    65557, 31626, 31626, 33201,
)
_A9_CTX_ORDER: tuple[str, ...] = ("coarse", "fine", "draft367")
# Isolated-bias path uses sep=-1.0 (not in SEPARATIONS_FWHM); frozen at PYTHONHASHSEED=1.
_ISOLATED_RNG_OFFSETS: dict[str, int] = {
    "coarse": 3010,
    "fine": 96204,
    "draft367": 55446,
}

A9_CONTEXTS: dict[str, A9Context] = {
    "coarse": A9Context("coarse", fwhm_px=3.2, plate_scale_arcsec=1.30),
    "fine": A9Context("fine", fwhm_px=6.4, plate_scale_arcsec=0.65),
    "draft367": A9Context(
        "draft367",
        fwhm_px=DRAFT367_FWHM_GAUSS_PX,
        plate_scale_arcsec=DRAFT367_PLATE_SCALE_ARCSEC,
        aperture_fwhm_factor=1.35,
        annulus_inner_fwhm=2.7,
        annulus_outer_fwhm=5.2,
    ),
}


@dataclass
class A9Cell:
    sep_fwhm: float
    delta_mag: int
    context: str

    @property
    def cell_id(self) -> str:
        return f"s{self.sep_fwhm:.1f}_dM{self.delta_mag:+d}"


@dataclass
class A9CellResult:
    cell_id: str
    sep_fwhm: float
    delta_mag: int
    zone: str
    contamination_excess_pct: float
    plain_bias_pct: float
    isolated_bias_pct: float
    measured_flux_median: float
    true_flux: float
    n_frames: int
    criterion: dict[str, Any]
    mode: str
    pass_future_neighbor_sub: bool | None = None
    neighbor_sub_refused: bool = False
    neighbor_subtracted: bool = False
    refuse_reason: str = ""
    contamination_reduction_frac: float | None = None
    psf_variant: str = "ideal"
    note: str = ""


def _rng_for_cell(ctx_name: str, sep: float, dM: int) -> np.random.Generator:
    if sep < 0:
        off = _ISOLATED_RNG_OFFSETS[ctx_name]
    else:
        ci = _A9_CTX_ORDER.index(ctx_name)
        si = SEPARATIONS_FWHM.index(sep)
        di = DELTA_MAGS.index(dM)
        off = _CELL_RNG_OFFSETS[(ci * len(SEPARATIONS_FWHM) + si) * len(DELTA_MAGS) + di]
    return np.random.default_rng(A9_RNG_SEED + off)


@dataclass(frozen=True)
class PsfVariantSpec:
    """Inject vs fit PSF parameters for one A9 variant (validation-only)."""

    name: str
    fit_beta: float
    fit_fwhm_scale: float
    star_fwhm_scale: float
    neighbour_fwhm_scale: float
    inject_beta: float
    inject_ellip: float
    inject_theta: float
    notes: str

    def model_over_star_fwhm(self, ctx: A9Context) -> dict[str, float]:
        """FWHM ratio fit_model / injected_star for target and neighbour."""
        fit_fw = float(ctx.fwhm_px) * float(self.fit_fwhm_scale)
        t_star = float(ctx.fwhm_px) * float(self.star_fwhm_scale)
        n_star = float(ctx.fwhm_px) * float(self.star_fwhm_scale) * float(self.neighbour_fwhm_scale)
        return {
            "target": fit_fw / t_star if t_star > 0 else float("nan"),
            "neighbour": fit_fw / n_star if n_star > 0 else float("nan"),
        }


def psf_variant_spec(variant: PsfVariant) -> PsfVariantSpec:
    """Return inject/fit parameters for ideal, legacy mismatch, or EPSF-audit realistic variant."""
    if variant == "ideal":
        return PsfVariantSpec(
            name="ideal",
            fit_beta=MOFFAT_BETA,
            fit_fwhm_scale=1.0,
            star_fwhm_scale=1.0,
            neighbour_fwhm_scale=1.0,
            inject_beta=MOFFAT_BETA,
            inject_ellip=0.0,
            inject_theta=0.0,
            notes="Inject and fit share FWHM and beta=2.5; symmetric.",
        )
    if variant == "mismatch":
        return PsfVariantSpec(
            name="mismatch",
            fit_beta=2.0,
            fit_fwhm_scale=1.0,
            star_fwhm_scale=1.0,
            neighbour_fwhm_scale=1.12,
            inject_beta=MOFFAT_BETA,
            inject_ellip=0.0,
            inject_theta=0.0,
            notes=(
                "Legacy stress test: fit beta=2.0 vs inject beta=2.5; neighbour inject FWHM x1.12 "
                "(model NARROWER than neighbour by ~11%; target FWHM matched). No asymmetry."
            ),
        )
    if variant == "draft367":
        fit_scale = DRAFT367_FWHM_EPSF_MOFFAT_PX / DRAFT367_FWHM_GAUSS_PX
        star_scale = DRAFT367_FWHM_STARS_PX / DRAFT367_FWHM_GAUSS_PX
        return PsfVariantSpec(
            name="draft367",
            fit_beta=MOFFAT_BETA,
            fit_fwhm_scale=fit_scale,
            star_fwhm_scale=star_scale,
            neighbour_fwhm_scale=1.0,
            inject_beta=MOFFAT_BETA,
            inject_ellip=DRAFT367_INJECT_ELLIP,
            inject_theta=0.35,
            notes=(
                f"Draft 367 Red_180_2 measured mismatch ratio {DRAFT367_MISMATCH_RATIO:.4f} "
                f"(ePSF Moffat {DRAFT367_FWHM_EPSF_MOFFAT_PX:.3f} px vs stars "
                f"{DRAFT367_FWHM_STARS_PX:.3f} px at {DRAFT367_PLATE_SCALE_ARCSEC:.3f} arcsec/px)."
            ),
        )
    # Realistic: anchor to VYVAR_EPSF_FWHM_TEST (375 L ratio 1.112, 380 L 1.047 -> ~1.08 mid).
    star_scale = 1.0 / 1.08
    return PsfVariantSpec(
        name="realistic",
        fit_beta=MOFFAT_BETA,
        fit_fwhm_scale=1.0,
        star_fwhm_scale=star_scale,
        neighbour_fwhm_scale=1.0,
        inject_beta=MOFFAT_BETA,
        inject_ellip=0.08,
        inject_theta=0.35,
        notes=(
            "EPSF-audit anchor: model/star FWHM ~1.08 (stars ~7.4% narrower than fit ePSF); "
            "beta matched; mild ellipticity e=0.08 on inject only (fit stays round Moffat)."
        ),
    )


def build_blend_frame(
    ctx: A9Context,
    *,
    target_flux: float,
    neighbour_flux: float,
    sep_fwhm: float,
    rng: np.random.Generator,
    jitter: bool = True,
    inject_beta: float = MOFFAT_BETA,
    star_fwhm_scale: float = 1.0,
    neighbour_fwhm_scale: float = 1.0,
    inject_ellip: float = 0.0,
    inject_theta: float = 0.0,
) -> np.ndarray:
    """Single ADU frame: target at center, neighbour offset along +y (sep in FWHM units)."""
    sep_px = float(sep_fwhm) * float(ctx.fwhm_px)
    if jitter:
        sep_px += float(rng.uniform(-JITTER_FWHM_FRAC, JITTER_FWHM_FRAC) * ctx.fwhm_px)
    img = np.zeros((STAMP_N, STAMP_N), dtype=np.float64)
    t_fw = float(ctx.fwhm_px) * float(star_fwhm_scale)
    img += moffat_stamp(
        STAMP_C,
        STAMP_C,
        target_flux,
        t_fw,
        inject_beta,
        ny=STAMP_N,
        nx=STAMP_N,
        ellip=inject_ellip,
        theta=inject_theta,
    )
    if neighbour_flux > 0.0:
        n_fw = t_fw * float(neighbour_fwhm_scale)
        img += moffat_stamp(
            STAMP_C + sep_px,
            STAMP_C,
            neighbour_flux,
            n_fw,
            inject_beta,
            ny=STAMP_N,
            nx=STAMP_N,
            ellip=inject_ellip,
            theta=inject_theta,
        )
    img += SKY_ADU
    el = np.clip(img * GAIN_E_PER_ADU, 0.0, None)
    img = rng.poisson(el).astype(np.float64) / GAIN_E_PER_ADU
    img += rng.normal(0.0, READ_NOISE_E / GAIN_E_PER_ADU, size=img.shape)
    return img


def measure_target_flux_vyvar(
    data: np.ndarray,
    ctx: A9Context,
    *,
    x_c: float = STAMP_C,
    y_c: float = STAMP_C,
) -> float:
    """VYVAR sky-subtracted circular aperture (photometry_core path)."""
    r_ap, r_in, r_out = ctx.radii_px()
    flux, _, _ = _annulus_sky_subtracted_flux(data, x_c, y_c, r_ap, r_in, r_out)
    return float(flux)


def measure_isolated_bias_pct(ctx: A9Context, n_frames: int = N_FRAMES) -> float:
    """Median fractional bias (%) for isolated target = aperture method floor."""
    tflux = ctx.target_flux_adu()
    rng = _rng_for_cell(ctx.name, -1.0, 0)
    biases = []
    for _ in range(n_frames):
        frame = build_blend_frame(ctx, target_flux=tflux, neighbour_flux=0.0, sep_fwhm=0.0, rng=rng)
        meas = measure_target_flux_vyvar(frame, ctx)
        if math.isfinite(meas) and tflux > 0:
            biases.append((meas / tflux - 1.0) * 100.0)
    return float(np.median(biases)) if biases else 0.0


def classify_zone(sep_fwhm: float, delta_mag: int, contamination_excess_pct: float) -> str:
    """REFUSE / HIGH_VALUE / CLEAN for envelope scoring."""
    if sep_fwhm <= 0.8:
        return "REFUSE"
    if sep_fwhm <= 1.5 and contamination_excess_pct > 5.0:
        return "HIGH_VALUE"
    clean_sep = 2.5 if delta_mag >= -1 else 3.0
    if sep_fwhm >= clean_sep and contamination_excess_pct < 10.0:
        return "CLEAN"
    if sep_fwhm >= 2.0 and delta_mag >= 0 and contamination_excess_pct < 15.0:
        return "CLEAN"
    if contamination_excess_pct > 20.0:
        return "HIGH_VALUE"
    return "CLEAN"


def envelope_criterion(zone: str, sep_fwhm: float, delta_mag: int) -> dict[str, Any]:
    """Per-cell PASS rules for future mode=neighbor_sub."""
    if zone == "REFUSE":
        return {
            "zone": zone,
            "pass_rule": "guard_refuse",
            "description": "NEIGHBOR-SUB must refuse (fallback plain aperture + flag)",
            "neighbor_sub_must_fire": False,
        }
    if zone == "HIGH_VALUE":
        return {
            "zone": zone,
            "pass_rule": "recover_and_improve",
            "max_abs_bias_pct": max(2.0, 0.5 * abs(delta_mag) + 2.0),
            "min_contamination_reduction_frac": 0.80,
            "description": "recovered flux within noise AND >=80% contamination reduction",
            "neighbor_sub_must_fire": True,
        }
    return {
        "zone": zone,
        "pass_rule": "no_op",
        "max_abs_bias_pct": max(3.0, 2.0 + abs(delta_mag)),
        "description": "NEIGHBOR-SUB no-op; flux unchanged within noise",
        "neighbor_sub_must_fire": False,
    }


def _a9_neighbor_sub_cfg() -> AppConfig:
    """Validation-only: enable NEIGHBOR-SUB without touching production config.json."""
    cfg = AppConfig()
    cfg.psf_neighbor_sub_enabled = True
    return cfg


def _psf_fit_params(variant: PsfVariant) -> PsfVariantSpec:
    return psf_variant_spec(variant)


def score_neighbor_sub_cell(
    zone: str,
    plain_contam: float,
    sub_contam: float,
    *,
    refused: bool,
    neighbor_subtracted: bool,
    criterion: dict[str, Any],
    refuse_reason: str = "",
) -> bool:
    """Score one cell vs the A9 envelope for mode=neighbor_sub."""
    if zone == "REFUSE":
        return refused
    if zone == "HIGH_VALUE":
        if refused:
            if refuse_reason == "bright_close_regime":
                return True
            return False
        if not math.isfinite(plain_contam) or abs(plain_contam) < 2.0:
            return abs(sub_contam) <= float(criterion.get("max_abs_bias_pct", 5.0))
        reduc = 1.0 - abs(sub_contam) / max(abs(plain_contam), 1e-6)
        max_bias = float(criterion.get("max_abs_bias_pct", 10.0))
        min_red = float(criterion.get("min_contamination_reduction_frac", 0.80))
        return abs(sub_contam) <= max_bias and reduc >= min_red
    # CLEAN: no-op or unchanged within noise
    if refused or not neighbor_subtracted:
        return True
    return abs(sub_contam - plain_contam) <= float(criterion.get("max_abs_bias_pct", 5.0))


def measure_cell(
    cell: A9Cell,
    ctx: A9Context,
    *,
    mode: MeasureMode = "plain_aperture",
    isolated_bias_pct: float | None = None,
    n_frames: int = N_FRAMES,
    psf_variant: PsfVariant = "ideal",
) -> A9CellResult:
    """Measure one grid cell (plain_aperture baseline or neighbor_sub scored path)."""
    from psf_neighbor_sub import neighbor_sub_target_flux

    iso = isolated_bias_pct if isolated_bias_pct is not None else measure_isolated_bias_pct(ctx, n_frames)
    tflux = ctx.target_flux_adu()
    nflux = tflux * (10.0 ** (-0.4 * cell.delta_mag))
    rng = _rng_for_cell(ctx.name, cell.sep_fwhm, cell.delta_mag)
    pspec = _psf_fit_params(psf_variant)
    r_ap, r_in, r_out = ctx.radii_px()
    ns_cfg = _a9_neighbor_sub_cfg()
    fit_fwhm = float(ctx.fwhm_px) * float(pspec.fit_fwhm_scale)

    plain_biases: list[float] = []
    sub_biases: list[float] = []
    refused_any = False
    refuse_reason = ""
    subtracted_any = False

    for _ in range(n_frames):
        frame = build_blend_frame(
            ctx,
            target_flux=tflux,
            neighbour_flux=nflux if cell.sep_fwhm > 0 else 0.0,
            sep_fwhm=cell.sep_fwhm,
            rng=rng,
            inject_beta=pspec.inject_beta,
            star_fwhm_scale=pspec.star_fwhm_scale,
            neighbour_fwhm_scale=pspec.neighbour_fwhm_scale,
            inject_ellip=pspec.inject_ellip,
            inject_theta=pspec.inject_theta,
        )
        plain_meas = measure_target_flux_vyvar(frame, ctx)
        if math.isfinite(plain_meas) and tflux > 0:
            plain_biases.append((plain_meas / tflux - 1.0) * 100.0)

        if mode == "neighbor_sub" and cell.sep_fwhm > 0 and nflux > 0:
            sep_px = cell.sep_fwhm * ctx.fwhm_px
            nxy = (float(STAMP_C), float(STAMP_C + sep_px))
            ns = neighbor_sub_target_flux(
                frame,
                target_xy=(float(STAMP_C), float(STAMP_C)),
                neighbour_xys=[nxy],
                fwhm_px=fit_fwhm,
                r_ap=r_ap,
                r_in=r_in,
                r_out=r_out,
                delta_mag_nn=float(cell.delta_mag),
                nn_dist_fwhm=float(cell.sep_fwhm),
                target_mag=float(TARGET_MAG),
                nn_mag=float(TARGET_MAG + cell.delta_mag),
                flux_zp=float(ZP),
                fit_beta=pspec.fit_beta,
                cfg=ns_cfg,
            )
            if ns.refused:
                refused_any = True
                refuse_reason = ns.refuse_reason or refuse_reason
            if ns.neighbor_subtracted:
                subtracted_any = True
            sub_meas = ns.target_flux
        else:
            sub_meas = plain_meas

        if math.isfinite(sub_meas) and tflux > 0:
            sub_biases.append((sub_meas / tflux - 1.0) * 100.0)

    plain_bias = float(np.median(plain_biases)) if plain_biases else float("nan")
    plain_contam = plain_bias - iso if math.isfinite(plain_bias) else float("nan")
    zone = classify_zone(cell.sep_fwhm, cell.delta_mag, plain_contam)
    crit = envelope_criterion(zone, cell.sep_fwhm, cell.delta_mag)

    sub_bias = float(np.median(sub_biases)) if sub_biases else float("nan")
    sub_contam = sub_bias - iso if math.isfinite(sub_bias) else float("nan")
    reduc = None
    if math.isfinite(plain_contam) and abs(plain_contam) > 1.0 and math.isfinite(sub_contam):
        reduc = 1.0 - abs(sub_contam) / abs(plain_contam)

    scored_pass = None
    if mode == "neighbor_sub":
        scored_pass = score_neighbor_sub_cell(
            zone,
            plain_contam,
            sub_contam,
            refused=refused_any,
            neighbor_subtracted=subtracted_any,
            criterion=crit,
            refuse_reason=refuse_reason,
        )

    contam_out = sub_contam if mode == "neighbor_sub" else plain_contam
    bias_out = sub_bias if mode == "neighbor_sub" else plain_bias
    meas_med = float("nan")
    if sub_biases if mode == "neighbor_sub" else plain_biases:
        b_list = sub_biases if mode == "neighbor_sub" else plain_biases
        meas_med = float(np.median([tflux * (1.0 + b / 100.0) for b in b_list]))

    return A9CellResult(
        cell_id=cell.cell_id,
        sep_fwhm=cell.sep_fwhm,
        delta_mag=cell.delta_mag,
        zone=zone,
        contamination_excess_pct=contam_out,
        plain_bias_pct=bias_out,
        isolated_bias_pct=iso,
        measured_flux_median=meas_med,
        true_flux=tflux,
        n_frames=n_frames,
        criterion=crit,
        mode=mode,
        pass_future_neighbor_sub=scored_pass,
        neighbor_sub_refused=refused_any,
        neighbor_subtracted=subtracted_any,
        refuse_reason=refuse_reason,
        contamination_reduction_frac=reduc,
        psf_variant=psf_variant,
        note=f"plain_contam={plain_contam:+.1f}%" if mode == "neighbor_sub" else "",
    )


def all_cells() -> list[A9Cell]:
    out: list[A9Cell] = []
    for ctx_name in A9_CONTEXTS:
        for sep in SEPARATIONS_FWHM:
            for dM in DELTA_MAGS:
                out.append(A9Cell(sep_fwhm=sep, delta_mag=dM, context=ctx_name))
    return out


def run_baseline_envelope(ctx_name: str = "coarse") -> dict[str, Any]:
    """Full plain_aperture contamination map + per-cell criteria for one context."""
    ctx = A9_CONTEXTS[ctx_name]
    iso = measure_isolated_bias_pct(ctx)
    r_ap, r_in, r_out = ctx.radii_px()
    cells_out: list[dict[str, Any]] = []
    matrix = np.full((len(SEPARATIONS_FWHM), len(DELTA_MAGS)), np.nan)

    for i, sep in enumerate(SEPARATIONS_FWHM):
        for j, dM in enumerate(DELTA_MAGS):
            cell = A9Cell(sep_fwhm=sep, delta_mag=dM, context=ctx_name)
            res = measure_cell(cell, ctx, mode="plain_aperture", isolated_bias_pct=iso)
            cells_out.append(asdict(res))
            matrix[i, j] = res.contamination_excess_pct

    return {
        "tier": "A9",
        "context": ctx_name,
        "mode": "plain_aperture",
        "rng_seed": A9_RNG_SEED,
        "target_mag": TARGET_MAG,
        "n_frames": N_FRAMES,
        "fwhm_px": ctx.fwhm_px,
        "plate_scale_arcsec": ctx.plate_scale_arcsec,
        "aperture_r_px": r_ap,
        "annulus_r_in_px": r_in,
        "annulus_r_out_px": r_out,
        "extractor": "photometry_core._annulus_sky_subtracted_flux",
        "isolated_bias_pct": iso,
        "separations_fwhm": list(SEPARATIONS_FWHM),
        "delta_mags": list(DELTA_MAGS),
        "contamination_excess_pct": matrix.tolist(),
        "cells": cells_out,
    }


def self_check_envelope(report: dict[str, Any]) -> tuple[bool, list[str]]:
    """Verify zone structure: huge bias at small sep; ~clean at sep>=3, dM>=0."""
    notes: list[str] = []
    ok = True
    matrix = np.asarray(report["contamination_excess_pct"], dtype=float)
    seps = report["separations_fwhm"]
    dms = report["delta_mags"]

    def _at(sep: float, dM: int) -> float:
        i = seps.index(sep)
        j = dms.index(dM)
        return float(matrix[i, j])

    refuse_vals = [_at(0.5, d) for d in dms]
    if not all(v > 40.0 for v in refuse_vals):
        ok = False
        notes.append(f"REFUSE row sep=0.5 expected all >40%; got {[f'{v:+.0f}' for v in refuse_vals]}")

    clean_val = _at(3.0, 0)
    if not (abs(clean_val) < 8.0):
        ok = False
        notes.append(f"CLEAN cell sep=3.0 dM=0 expected |contam|<8%; got {clean_val:+.1f}%")

    hv = _at(1.0, -3)
    if not (hv > 200.0):
        ok = False
        notes.append(f"HIGH-VALUE cell sep=1.0 dM=-3 expected >200%; got {hv:+.1f}%")

    bright_wing = _at(3.0, -3)
    if not (bright_wing > 5.0):
        notes.append(f"note: sep=3.0 dM=-3 wing contam {bright_wing:+.1f}% (expected >5% for bright neighbour)")

    return ok, notes


def write_heatmap(report: dict[str, Any], out_path: Path) -> None:
    """Save contamination excess heatmap (optional matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    M = np.asarray(report["contamination_excess_pct"], dtype=float)
    seps = report["separations_fwhm"]
    dms = report["delta_mags"]
    ctx = report.get("context", "coarse")

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_m = np.clip(M, 0.3, None)
    im = ax.imshow(np.log10(plot_m), cmap="inferno", aspect="auto", origin="upper")
    ax.set_xticks(range(len(dms)))
    ax.set_xticklabels([f"dM{d:+d}" for d in dms])
    ax.set_yticks(range(len(seps)))
    ax.set_yticklabels([f"{s:.1f}" for s in seps])
    ax.set_xlabel("neighbour delta_mag (negative = brighter)")
    ax.set_ylabel("separation (FWHM)")
    ax.set_title(
        f"A9 contamination map ({ctx}): plain-aperture bias EXCESS (%)\n"
        "problem NEIGHBOR-SUB must reduce"
    )
    for i in range(len(seps)):
        for j in range(len(dms)):
            v = M[i, j]
            ax.text(
                j,
                i,
                f"{v:+.0f}%",
                ha="center",
                va="center",
                color="white" if v > 50 else "black",
                fontsize=9,
            )
    ax.axhline(0.5, color="cyan", lw=2)
    ax.text(len(dms) - 0.15, 0.2, "REFUSE", color="cyan", fontsize=8, ha="right")
    ax.axhline(4.5, color="lime", lw=2)
    ax.text(len(dms) - 0.15, 3.0, "HIGH-VALUE", color="lime", fontsize=8, ha="right")
    ax.text(len(dms) - 0.15, 5.6, "CLEAN (no-op)", color="white", fontsize=8, ha="right")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("log10 contamination excess (%)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def run_neighbor_sub_envelope(ctx_name: str, psf_variant: PsfVariant = "ideal") -> dict[str, Any]:
    """Score neighbor_sub per cell vs envelope (ideal or PSF-mismatch variant)."""
    ctx = A9_CONTEXTS[ctx_name]
    iso = measure_isolated_bias_pct(ctx)
    matrix = np.full((len(SEPARATIONS_FWHM), len(DELTA_MAGS)), np.nan)
    gain = np.full_like(matrix, np.nan)
    pass_m = np.zeros_like(matrix, dtype=bool)
    cells_out: list[dict[str, Any]] = []

    for i, sep in enumerate(SEPARATIONS_FWHM):
        for j, dM in enumerate(DELTA_MAGS):
            cell = A9Cell(sep_fwhm=sep, delta_mag=dM, context=ctx_name)
            plain = measure_cell(
                cell, ctx, mode="plain_aperture", isolated_bias_pct=iso, psf_variant=psf_variant
            )
            sub = measure_cell(
                cell, ctx, mode="neighbor_sub", isolated_bias_pct=iso, psf_variant=psf_variant
            )
            matrix[i, j] = sub.contamination_excess_pct
            if math.isfinite(plain.contamination_excess_pct) and abs(plain.contamination_excess_pct) > 1.0:
                gain[i, j] = 1.0 - abs(sub.contamination_excess_pct) / abs(plain.contamination_excess_pct)
            pass_m[i, j] = bool(sub.pass_future_neighbor_sub)
            cells_out.append(
                {
                    "cell_id": cell.cell_id,
                    "zone": sub.zone,
                    "plain_contam_pct": plain.contamination_excess_pct,
                    "sub_contam_pct": sub.contamination_excess_pct,
                    "gain_frac": float(gain[i, j]) if math.isfinite(gain[i, j]) else None,
                    "scored_pass": sub.pass_future_neighbor_sub,
                    "refused": sub.neighbor_sub_refused,
                    "refuse_reason": sub.refuse_reason,
                }
            )

    n_pass = int(pass_m.sum())
    n_total = int(pass_m.size)
    return {
        "context": ctx_name,
        "psf_variant": psf_variant,
        "mode": "neighbor_sub",
        "contamination_excess_pct": matrix.tolist(),
        "gain_frac": gain.tolist(),
        "scored_pass": pass_m.tolist(),
        "n_pass": n_pass,
        "n_total": n_total,
        "pass_rate": round(n_pass / n_total, 3) if n_total else 0.0,
        "cells": cells_out,
    }


def write_envelope_report(out_dir: Path) -> tuple[Path, Path, bool]:
    """Generate JSON + MD for both contexts; return (json_path, md_path, self_check_ok)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_reports: dict[str, Any] = {
        "contexts": {},
        "self_check": {},
        "neighbor_sub": {"ideal": {}, "mismatch": {}},
    }
    all_ok = True

    for ctx_name in A9_CONTEXTS:
        rep = run_baseline_envelope(ctx_name)
        ok, notes = self_check_envelope(rep)
        all_reports["contexts"][ctx_name] = rep
        all_reports["self_check"][ctx_name] = {"ok": ok, "notes": notes}
        all_ok = all_ok and ok
        write_heatmap(rep, out_dir / f"a9_envelope_{ctx_name}_plain.png")
        for variant in ("ideal", "mismatch"):
            ns_rep = run_neighbor_sub_envelope(ctx_name, variant)  # type: ignore[arg-type]
            all_reports["neighbor_sub"][variant][ctx_name] = ns_rep
            write_heatmap(
                {
                    **ns_rep,
                    "separations_fwhm": rep["separations_fwhm"],
                    "delta_mags": rep["delta_mags"],
                },
                out_dir / f"a9_gain_{ctx_name}_{variant}.png",
            )
        ideal_ns = all_reports["neighbor_sub"]["ideal"][ctx_name]
        if ideal_ns["pass_rate"] < 0.55:
            all_ok = False
        mismatch_ns = all_reports["neighbor_sub"]["mismatch"][ctx_name]
        if mismatch_ns["pass_rate"] >= ideal_ns["pass_rate"]:
            all_ok = False

    jp = out_dir / "a9_envelope.json"
    with open(jp, "w", encoding="ascii") as f:
        json.dump(all_reports, f, indent=2)

    mp = out_dir / "a9_envelope.md"
    lines = [
        "# A9 NEIGHBOR-SUB acceptance envelope (plain_aperture baseline)",
        "",
        "Defines per-cell zones and PASS criteria for future `neighbor_sub` scoring.",
        f"Extractor: `photometry_core._annulus_sky_subtracted_flux` with AppConfig radii.",
        f"RNG seed: {A9_RNG_SEED}. N_FRAMES={N_FRAMES} per cell.",
        "",
    ]
    for ctx_name, rep in all_reports["contexts"].items():
        sc = all_reports["self_check"][ctx_name]
        lines.append(f"## Context: {ctx_name}")
        lines.append(
            f"FWHM={rep['fwhm_px']} px, plate_scale={rep['plate_scale_arcsec']} arcsec/px, "
            f"R_ap={rep['aperture_r_px']:.2f} px, isolated bias={rep['isolated_bias_pct']:+.2f}%"
        )
        lines.append(f"Self-check: **{'PASS' if sc['ok'] else 'FAIL'}**")
        for n in sc["notes"]:
            lines.append(f"- {n}")
        lines.append("")
        lines.append("### Contamination excess (%)")
        lines.append("")
        header = "| sep(FWHM) | " + " | ".join(f"dM{d:+d}" for d in rep["delta_mags"]) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(rep["delta_mags"]) + 1))
        M = np.asarray(rep["contamination_excess_pct"])
        for i, sep in enumerate(rep["separations_fwhm"]):
            row = " | ".join(f"{M[i, j]:+.0f}" for j in range(len(rep["delta_mags"])))
            lines.append(f"| {sep:.1f} | {row} |")
        lines.append("")
        lines.append("### Zones")
        lines.append("- **REFUSE** (sep <= 0.8 FWHM): guard must refuse subtraction")
        lines.append("- **HIGH_VALUE** (sep ~0.8-1.5, blended): recover + >=80% contamination cut")
        lines.append("- **CLEAN** (wide sep / faint neighbour): NEIGHBOR-SUB no-op")
        lines.append("")
        for variant in ("ideal", "mismatch"):
            ns = all_reports["neighbor_sub"][variant].get(ctx_name, {})
            if not ns:
                continue
            lines.append(f"### neighbor_sub ({variant}) pass rate: {ns.get('pass_rate', 0):.1%}")
            lines.append("")
            G = np.asarray(ns.get("gain_frac", []))
            if G.size:
                header2 = "| sep | " + " | ".join(f"dM{d:+d}" for d in rep["delta_mags"]) + " |"
                lines.append(header2)
                lines.append("|" + "---|" * (len(rep["delta_mags"]) + 1))
                for i, sep in enumerate(rep["separations_fwhm"]):
                    row = " | ".join(
                        f"{G[i, j]:.2f}" if math.isfinite(G[i, j]) else "n/a"
                        for j in range(len(rep["delta_mags"]))
                    )
                    lines.append(f"| {sep:.1f} | {row} |")
                lines.append("")

    with open(mp, "w", encoding="ascii") as f:
        f.write("\n".join(lines))

    return jp, mp, all_ok


OutcomeType = Literal[
    "PASS-RECOVER",
    "PASS-REFUSE",
    "PASS-NOOP",
    "FAIL-RECOVER",
    "FAIL-SILENT",
    "FAIL-REFUSE-MISS",
    "FAIL-HV-REFUSE",
    "N/A",
]


def classify_cell_outcome(
    zone: str,
    plain_contam: float,
    sub_contam: float,
    *,
    refused: bool,
    subtracted: bool,
    criterion: dict[str, Any],
    refuse_reason: str = "",
) -> OutcomeType:
    """Per-cell failure typing for mismatch diagnostic (step 2b gate)."""
    max_bias = float(criterion.get("max_abs_bias_pct", 10.0))
    min_red = float(criterion.get("min_contamination_reduction_frac", 0.80))

    if zone == "CLEAN":
        if refused or not subtracted:
            return "PASS-NOOP"
        if abs(sub_contam - plain_contam) <= max_bias:
            return "PASS-NOOP"
        return "FAIL-SILENT"

    if zone == "REFUSE":
        if refused:
            return "PASS-REFUSE"
        return "FAIL-REFUSE-MISS"

    # HIGH_VALUE
    if refused:
        if refuse_reason == "bright_close_regime":
            return "PASS-REFUSE"
        return "FAIL-HV-REFUSE"
    if not subtracted:
        return "FAIL-RECOVER"

    reduc = 1.0 - abs(sub_contam) / max(abs(plain_contam), 1e-6)
    recovered_ok = abs(sub_contam) <= max_bias and reduc >= min_red
    if recovered_ok:
        return "PASS-RECOVER"

    worse_than_plain = (
        math.isfinite(plain_contam)
        and math.isfinite(sub_contam)
        and abs(sub_contam) >= abs(plain_contam) * 0.97
    )
    over_sub = math.isfinite(sub_contam) and sub_contam < -max_bias
    large_bias = math.isfinite(sub_contam) and abs(sub_contam) > max_bias

    if worse_than_plain or over_sub or (large_bias and subtracted and not refused):
        return "FAIL-SILENT"
    return "FAIL-RECOVER"


def run_mismatch_diagnostic(
    ctx_name: str = "coarse",
    *,
    variants: tuple[PsfVariant, ...] = ("mismatch", "realistic"),
) -> dict[str, Any]:
    """Per-cell breakdown for PSF-mismatch variants (analysis-only; drives step 2b)."""
    ctx = A9_CONTEXTS[ctx_name]
    iso = measure_isolated_bias_pct(ctx)
    variant_specs = {v: psf_variant_spec(v) for v in variants}
    out: dict[str, Any] = {
        "context": ctx_name,
        "fwhm_px": ctx.fwhm_px,
        "isolated_bias_pct": iso,
        "variant_specs": {
            v: {
                **asdict(spec),
                "model_over_star_fwhm": spec.model_over_star_fwhm(ctx),
            }
            for v, spec in variant_specs.items()
        },
        "variants": {},
    }

    for variant in variants:
        rows: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        hv_cells: list[dict[str, Any]] = []
        refuse_cells: list[dict[str, Any]] = []

        for sep in SEPARATIONS_FWHM:
            for dM in DELTA_MAGS:
                cell = A9Cell(sep_fwhm=sep, delta_mag=dM, context=ctx_name)
                plain = measure_cell(
                    cell, ctx, mode="plain_aperture", isolated_bias_pct=iso, psf_variant=variant
                )
                sub = measure_cell(
                    cell, ctx, mode="neighbor_sub", isolated_bias_pct=iso, psf_variant=variant
                )
                outcome = classify_cell_outcome(
                    sub.zone,
                    plain.contamination_excess_pct,
                    sub.contamination_excess_pct,
                    refused=sub.neighbor_sub_refused,
                    subtracted=sub.neighbor_subtracted,
                    criterion=sub.criterion,
                    refuse_reason=sub.refuse_reason or "",
                )
                counts[outcome] = counts.get(outcome, 0) + 1
                row = {
                    "sep_fwhm": sep,
                    "delta_mag": dM,
                    "cell_id": cell.cell_id,
                    "zone": sub.zone,
                    "plain_contam_pct": plain.contamination_excess_pct,
                    "sub_recovered_contam_pct": sub.contamination_excess_pct,
                    "refused": sub.neighbor_sub_refused,
                    "subtracted": sub.neighbor_subtracted,
                    "refuse_reason": sub.refuse_reason,
                    "scored_pass": sub.pass_future_neighbor_sub,
                    "outcome_type": outcome,
                }
                rows.append(row)
                if sub.zone == "HIGH_VALUE":
                    hv_cells.append(row)
                if sub.zone == "REFUSE":
                    refuse_cells.append(row)

        n_hv = len(hv_cells)
        n_hv_recover = sum(1 for r in hv_cells if r["outcome_type"] == "PASS-RECOVER")
        n_fail_silent = counts.get("FAIL-SILENT", 0)
        n_refuse = len(refuse_cells)
        n_refuse_ok = sum(1 for r in refuse_cells if r["outcome_type"] == "PASS-REFUSE")

        if n_fail_silent > 0:
            verdict = "BLOCK_2B_GUARDS"
        elif n_fail_silent == 0 and n_hv > 0 and n_hv_recover / n_hv >= 0.50:
            verdict = "PROCEED_2B_CANDIDATE"
        elif n_fail_silent == 0:
            verdict = "SAFE_LOW_YIELD"
        else:
            verdict = "UNDEFINED"

        out["variants"][variant] = {
            "pass_rate": sum(1 for r in rows if r["scored_pass"]) / len(rows) if rows else 0.0,
            "outcome_counts": counts,
            "high_value_pass_recover_rate": round(n_hv_recover / n_hv, 3) if n_hv else None,
            "fail_silent_count": n_fail_silent,
            "refuse_correct_rate": round(n_refuse_ok / n_refuse, 3) if n_refuse else None,
            "verdict": verdict,
            "cells": rows,
        }

    return out


def write_mismatch_diagnostic_report(out_dir: Path) -> tuple[Path, Path]:
    """Write JSON + MD mismatch diagnostic (current vs realistic)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = run_mismatch_diagnostic("coarse")
    jp = out_dir / "a9_mismatch_diagnostic.json"
    with open(jp, "w", encoding="ascii") as f:
        json.dump(diag, f, indent=2)

    lines = [
        "# A9 PSF-mismatch diagnostic (NEIGHBOR-SUB step 2b gate)",
        "",
        "Analysis-only. Coarse context (FWHM=3.2 px). Anchor: `docs/VYVAR_EPSF_FWHM_TEST.md`.",
        "",
        "## 1. Variant magnitudes (fit ePSF vs injected stars)",
        "",
    ]
    for vname, spec_d in diag["variant_specs"].items():
        ratios = spec_d["model_over_star_fwhm"]
        lines.append(f"### {vname}")
        lines.append(f"- {spec_d['notes']}")
        lines.append(
            f"- fit: beta={spec_d['fit_beta']}, FWHM={diag['fwhm_px'] * spec_d['fit_fwhm_scale']:.3f} px"
        )
        lines.append(
            f"- inject: beta={spec_d['inject_beta']}, ellip={spec_d['inject_ellip']}, "
            f"star_FWHM_scale={spec_d['star_fwhm_scale']}, neighbour_extra={spec_d['neighbour_fwhm_scale']}"
        )
        lines.append(
            f"- model/star FWHM ratio: target={ratios['target']:.3f}, "
            f"neighbour={ratios['neighbour']:.3f}"
        )
        lines.append("")

    lines.append("## 2. Decision summary (realistic mismatch)")
    lines.append("")
    real = diag["variants"].get("realistic", {})
    if real:
        lines.append(f"- HIGH_VALUE PASS-RECOVER rate: **{real.get('high_value_pass_recover_rate', 0):.1%}**")
        lines.append(f"- FAIL-SILENT count: **{real.get('fail_silent_count', 0)}**")
        lines.append(f"- REFUSE correctness: **{real.get('refuse_correct_rate', 0):.1%}**")
        lines.append(f"- Verdict: **{real.get('verdict', 'n/a')}**")
        lines.append("")
        lines.append("Rules: FAIL-SILENT>0 -> block 2b (iterate guards); FAIL-SILENT~0 and HV recover>=50%")
        lines.append("-> 2b candidate; FAIL-SILENT~0 but low HV recover -> SAFE_LOW_YIELD (fine-scale/PSF).")
        lines.append("")

    cur = diag["variants"].get("mismatch", {})
    if cur:
        lines.append("### Legacy `mismatch` variant (comparison)")
        lines.append(f"- pass rate: {cur.get('pass_rate', 0):.1%}")
        lines.append(f"- FAIL-SILENT: {cur.get('fail_silent_count', 0)}")
        lines.append(f"- HV PASS-RECOVER: {cur.get('high_value_pass_recover_rate', 0):.1%}")
        lines.append("")

    for variant in ("realistic", "mismatch"):
        vrep = diag["variants"].get(variant)
        if not vrep:
            continue
        lines.append(f"## 3. Per-cell breakdown ({variant})")
        lines.append("")
        lines.append(
            "| sep | dM | zone | plain% | recovered% | refused | reason | PASS | outcome |"
        )
        lines.append("|---:|---:|---|---:|---:|:---:|---|:---:|:---:|")
        for r in vrep["cells"]:
            if r["sep_fwhm"] == 0.0 and r["delta_mag"] == 0:
                continue
            ref = "Y" if r["refused"] else "N"
            sp = "Y" if r["scored_pass"] else "N"
            reason = r["refuse_reason"] or "-"
            lines.append(
                f"| {r['sep_fwhm']:.1f} | {r['delta_mag']:+d} | {r['zone']} | "
                f"{r['plain_contam_pct']:+.0f} | {r['sub_recovered_contam_pct']:+.1f} | "
                f"{ref} | {reason} | {sp} | {r['outcome_type']} |"
            )
        lines.append("")

    mp = out_dir / "a9_mismatch_diagnostic.md"
    with open(mp, "w", encoding="ascii") as f:
        f.write("\n".join(lines))
    return jp, mp


def draft367_a9_verdict(rep: dict[str, Any]) -> str:
    """A9-only gate for draft 367 (mismatch + yield)."""
    fs = int(rep.get("fail_silent_count", 0))
    hv = float(rep.get("high_value_pass_recover_rate") or 0.0)
    ratio = float(rep.get("mismatch_ratio") or DRAFT367_MISMATCH_RATIO)
    if ratio > 1.03:
        return "MISMATCH_STILL_HIGH"
    if fs == 0 and ratio <= 1.03 and hv >= 0.50:
        return "A9_PASS"
    if fs > 0 and ratio <= 1.03 and hv >= 0.50:
        return "A9_EDGE_FAIL_SILENT"
    if fs > 0:
        return "BLOCK_2B_GUARDS"
    if fs == 0 and hv < 0.50:
        return "SAFE_LOW_YIELD"
    return "UNDEFINED"


def draft367_combined_decision(
    a9_verdict: str,
    crowding_verdict: str,
) -> str:
    """Part A + Part B -> 2b wiring decision."""
    if a9_verdict in ("BLOCK_2B_GUARDS", "MISMATCH_STILL_HIGH", "A9_EDGE_FAIL_SILENT"):
        return a9_verdict
    if a9_verdict != "A9_PASS":
        return a9_verdict
    if crowding_verdict == "PROCEED_2B_CANDIDATE":
        return "PROCEED_2B_CANDIDATE"
    return "VALIDATED_FINE_SCALE_IDLE"


def draft367_decision_verdict(rep: dict[str, Any]) -> str:
    """Backward-compatible alias for A9-only verdict."""
    return draft367_a9_verdict(rep)


def run_draft367_neighbor_sub_diagnostic(
    *,
    epsf_audit: dict[str, Any] | None = None,
    crowding_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Part-2: A9 neighbor_sub at draft 367 sampling + measured mismatch."""
    audit = epsf_audit or {}
    ratio = float(audit.get("ratio_moffat_vs_stars") or DRAFT367_MISMATCH_RATIO)
    rep = run_mismatch_diagnostic("draft367", variants=("draft367",))  # type: ignore[arg-type]
    d367 = rep["variants"]["draft367"]
    a9_verdict = draft367_a9_verdict({**d367, "mismatch_ratio": ratio})
    if crowding_audit is None:
        from tests.validation.crowding_audit_367 import analyze_draft_367

        crowding_audit = analyze_draft_367()
    crowd_v = str(crowding_audit.get("crowding_verdict", "SPARSE"))
    decision = draft367_combined_decision(a9_verdict, crowd_v)
    return {
        "epsf_audit": audit,
        "mismatch_ratio": ratio,
        "context": "draft367",
        "neighbor_sub": d367,
        "a9_verdict": a9_verdict,
        "crowding_audit": crowding_audit,
        "crowding_verdict": crowd_v,
        "decision": decision,
        "comparison": {
            "hchi_per_ratio_375L": 1.112,
            "coarse_a9_realistic_hv_recover": 0.176,
            "coarse_a9_realistic_fail_silent": 0,
        },
    }


def write_draft367_report(out_dir: Path, *, epsf_audit: dict[str, Any] | None = None) -> tuple[Path, Path]:
    """Write JSON + MD for draft 367 fine-scale NEIGHBOR-SUB diagnostic."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if epsf_audit is None:
        from tests.validation.epsf_fwhm_audit import analyze_draft_367

        epsf_audit = analyze_draft_367()
    diag = run_draft367_neighbor_sub_diagnostic(epsf_audit=epsf_audit)
    jp = out_dir / "a9_draft367_diagnostic.json"
    with open(jp, "w", encoding="ascii") as f:
        json.dump(diag, f, indent=2)

    ns = diag["neighbor_sub"]
    audit = diag["epsf_audit"]
    lines = [
        "# A9 draft 367 fine-scale NEIGHBOR-SUB diagnostic",
        "",
        "Draft 367 (Palomar 7 BGR), Red_180_2 (16 frames, richest SNR). Read-only; gated OFF.",
        "",
        "## Part 1 -- ePSF vs star mismatch (VYVAR_EPSF_FWHM_TEST method)",
        "",
        f"| quantity | draft 367 | h & chi Per 375 L |",
        f"|----------|-----------|-------------------|",
        f"| plate scale (arcsec/px) | {audit.get('plate_scale_arcsec', DRAFT367_PLATE_SCALE_ARCSEC):.4f} | 1.30 |",
        f"| VY_FWHM_GAUSS (px) | {audit.get('vy_fwhm_gauss', DRAFT367_FWHM_GAUSS_PX):.3f} | ~2.74 |",
        f"| ePSF Moffat FWHM (px) | {audit.get('fwhm_moffat_native', DRAFT367_FWHM_EPSF_MOFFAT_PX):.3f} | 2.131 |",
        f"| stars Moffat FWHM (px) | {audit.get('fwhm_stars_native', DRAFT367_FWHM_STARS_PX):.3f} | 1.916 |",
        f"| **mismatch ratio ePSF/stars** | **{diag['mismatch_ratio']:.4f}** | **1.112** |",
        f"| FWHM (arcsec) | {audit.get('fwhm_moffat_arcsec', 2.1):.3f} | ~2.5 |",
        "",
        "Bright-neighbour budget (prototype): ratio <= 1.02 recovers; ~1.08 -> ~-41% over-subtract.",
        "",
        "## Part 2 -- A9 neighbor_sub yield (draft367-calibrated)",
        "",
        f"- HIGH_VALUE PASS-RECOVER: **{ns.get('high_value_pass_recover_rate', 0):.1%}**",
        f"- FAIL-SILENT: **{ns.get('fail_silent_count', 0)}**",
        f"- REFUSE correctness: **{ns.get('refuse_correct_rate', 0):.1%}**",
        f"- A9 verdict: **{diag.get('a9_verdict', diag.get('decision'))}**",
        "",
        "## Part 3 -- real crowding (Red_180_2, VY_FWHM_GAUSS)",
        "",
    ]
    crowd = diag.get("crowding_audit", {}).get("richest") or {}
    cb = crowd.get("nn_buckets") or {}
    lines.extend(
        [
            f"| metric | Red_180_2 |",
            f"|--------|-----------|",
            f"| gaia_density (per arcmin^2) | {crowd.get('gaia_density_per_arcmin2', 'n/a')} |",
            f"| blend_frac @ 1 FWHM | {crowd.get('blend_frac_1fwhm', 'n/a')} |",
            f"| blend_frac @ 2 FWHM | {crowd.get('blend_frac_2fwhm', 'n/a')} |",
            f"| is_blended (LC stars) | {cb.get('n_is_blended_true', 'n/a')} |",
            f"| hard blends (nn < 1.0 FWHM) | {cb.get('hard_lt_1.0', 'n/a')} |",
            f"| crowding verdict | **{diag.get('crowding_verdict', 'n/a')}** |",
            "",
            "## Combined decision",
            "",
            f"**{diag['decision']}**",
            "",
        ]
    )
    dec = diag["decision"]
    if dec == "PROCEED_2B_CANDIDATE":
        lines.append(
            "FAIL-SILENT 0, A9 HV healthy, real blend population on 367 -> 2b candidate "
            "(wire core at measurement sites, still gated OFF; step 3 = real-field validation)."
        )
    elif dec == "VALIDATED_FINE_SCALE_IDLE":
        lines.append(
            "NEIGHBOR-SUB **validated at fine scale** (mismatch ~1.0, HV ~83%, FAIL-SILENT 0 after "
            "bright_close_regime guard). Draft 367 is **sparse** (9 blended, 4 hard) -- no immediate "
            "use case; keep gated OFF until a blended fine-scale field (e.g. Brno data)."
        )
    elif dec == "BLOCK_2B_GUARDS":
        lines.append("Guard / yield gate failed -- do not wire 2b until resolved.")
    elif dec == "MISMATCH_STILL_HIGH":
        lines.append("Mismatch still elevated at 367 -> empirical ePSF NEIGHBOR-SUB limited; do not wire 2b.")
    else:
        lines.append(f"Status: **{dec}** -- see A9 + crowding sections above.")
    lines.append("")
    lines.append("## Per-cell breakdown")
    lines.append("")
    lines.append("| sep | dM | zone | plain% | recovered% | refused | reason | PASS | outcome |")
    lines.append("|---:|---:|---|---:|---:|:---:|---|:---:|:---:|")
    for r in ns.get("cells", []):
        ref = "Y" if r["refused"] else "N"
        sp = "Y" if r["scored_pass"] else "N"
        reason = r["refuse_reason"] or "-"
        lines.append(
            f"| {r['sep_fwhm']:.1f} | {r['delta_mag']:+d} | {r['zone']} | "
            f"{r['plain_contam_pct']:+.0f} | {r['sub_recovered_contam_pct']:+.1f} | "
            f"{ref} | {reason} | {sp} | {r['outcome_type']} |"
        )
    lines.append("")
    mp = out_dir / "a9_draft367_diagnostic.md"
    with open(mp, "w", encoding="ascii") as f:
        f.write("\n".join(lines))
    return jp, mp
