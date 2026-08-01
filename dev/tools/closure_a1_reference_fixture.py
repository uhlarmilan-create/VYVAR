#!/usr/bin/env python3
"""VYVAR closure A-1: independent reference fixture for the differential-aperture metric.

Purpose
-------
Any harness that computes ``delta_ap`` must reproduce this file's numbers before its
output on real data is admissible. The fixture needs no VYVAR code, no observatory data
and no network: it renders a known PSF, measures it with photutils, and states the answer
in explicit millimagnitudes.

It exists because three consecutive closure steps produced a decisive number that was
wrong (2.69, then 0.203), and every self-test used was invariant to a global scale error.
This fixture is NOT scale-invariant: it checks absolute values against an independently
computed expectation.

Layers
------
L1  closed-form Moffat enclosed energy                      -- analytic
L2  synthetic pixel-integrated image + photutils exact apertures at the TRUE centroid
L3  integer-centre + hard ``dist <= r`` mask (the Step 1b method) -- for bias measurement

L1 and L2 differ by 2-3 percentage points at r ~ 1.9 px. That difference is pixel
integration and is physical. L2 is the reference for anything compared against measured
data; L1 alone is not.

Verified environment: photutils 3.0.0, astropy 8.0.1, numpy 2.x (VYVAR's pinned versions).

Usage
-----
    python dev/tools/closure_a1_reference_fixture.py            # run all gates
    python dev/tools/closure_a1_reference_fixture.py --emit     # print JSON of expected values
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry

# ---------------------------------------------------------------- configuration
BETA = 3.0
SKY_ADU = 1570.0          # anchor draft_435 measured sky
TOTAL_FLUX = 2.0e5
OVERSAMPLE = 21           # convergence verified: 11 -> 21 changes EE by 7e-5
COG_RADII = np.arange(0.25, 12.0001, 0.25)
NORM_RADIUS = 12.0
R_TARGET = 1.916          # anchor r_min_px clamp
# Sky annulus for COG / EE measurement (Step 1l L4):
#   r_in  = max(r_norm + 2.0, 4.75 * fw) = 14.0 px  (must exceed NORM_RADIUS)
#   r_out = max(r_in + 0.5, 9.0 * fw)  = 21.555 px
# Prior geometries: 25-45 px (Step 1b harness); 11.376-21.555 px (Step 1k, overlapped r_norm).
FW_PROD = 2.395
ANN_INNER_FWHM = 4.75
ANN_OUTER_FWHM = 9.0
_r_in_prod = max(NORM_RADIUS + 2.0, ANN_INNER_FWHM * FW_PROD)
ANNULUS_IN = _r_in_prod
ANNULUS_OUT = max(_r_in_prod + 0.5, ANN_OUTER_FWHM * FW_PROD)

SUBSETS = {
    "G_8_9":   [3.166, 3.016, 2.866, 2.866],
    "G_9_11":  [2.866, 2.816, 2.666, 2.516, 2.416],
    "G_gt_11": [2.266, 2.116, 2.016, 1.916, 1.916],
}
R50_GRID = [1.46, 1.60, 1.75, 1.87, 1.97]   # span reported on the anchor

TOL_MMAG = 1.0            # absolute agreement required against the emitted table


# ---------------------------------------------------------------- L1 closed form
def ee_moffat(r: float, alpha: float, beta: float) -> float:
    return 1.0 - (1.0 + (r / alpha) ** 2) ** (1.0 - beta)


def alpha_from_r50(r50: float, beta: float) -> float:
    return r50 / math.sqrt(2.0 ** (1.0 / (beta - 1.0)) - 1.0)


def fwhm_from_r50(r50: float, beta: float) -> float:
    a = alpha_from_r50(r50, beta)
    return 2.0 * a * math.sqrt(2.0 ** (1.0 / beta) - 1.0)


# ---------------------------------------------------------------- L2 synthetic + photutils
def render_moffat(shape, xc, yc, alpha, beta, total_flux, sky, oversample=OVERSAMPLE):
    ny, nx = shape
    o = oversample
    yy, xx = np.mgrid[0 : ny * o, 0 : nx * o]
    xs = (xx + 0.5) / o - 0.5
    ys = (yy + 0.5) / o - 0.5
    rr2 = (xs - xc) ** 2 + (ys - yc) ** 2
    norm = (beta - 1.0) / (math.pi * alpha**2)
    fine = total_flux * norm * (1.0 + rr2 / alpha**2) ** (-beta)
    return fine.reshape(ny, o, nx, o).mean(axis=(1, 3)) + sky


def ee_curve_photutils(img, xc, yc, radii=COG_RADII):
    ann = CircularAnnulus([(xc, yc)], r_in=ANNULUS_IN, r_out=ANNULUS_OUT)
    sky_pp = float(aperture_photometry(img, ann)["aperture_sum"][0] / ann.area)
    flux = []
    for r in radii:
        ap = CircularAperture([(xc, yc)], r=float(r))
        s = float(aperture_photometry(img, ap)["aperture_sum"][0])
        flux.append(s - sky_pp * ap.area)
    arr = np.asarray(flux, dtype=np.float64)
    return arr / arr[-1]


# ---------------------------------------------------------------- L3 harness method
def ee_curve_integer_centre(img, xc, yc, radii=COG_RADII):
    """Reproduces dev/tools/closure_step1b_differential_aperture.py::_curve_of_growth."""
    xi, yi = int(round(xc)), int(round(yc))
    max_r = int(math.ceil(float(np.max(radii)) + ANNULUS_OUT + 2))
    yy, xx = np.mgrid[yi - max_r : yi + max_r + 1, xi - max_r : xi + max_r + 1]
    patch = img[yi - max_r : yi + max_r + 1, xi - max_r : xi + max_r + 1]
    dist = np.hypot(xx - xi, yy - yi)
    sky = float(np.median(patch[(dist >= ANNULUS_IN) & (dist <= ANNULUS_OUT)]))
    flux = [max(float(np.sum((patch - sky)[dist <= r])), 0.0) for r in radii]
    arr = np.asarray(flux, dtype=np.float64)
    return arr / (arr[-1] if arr[-1] > 0 else 1.0)


# ---------------------------------------------------------------- the metric
def ee_at(radii, ee, r):
    return float(np.interp(r, radii, ee))


def delta_ap_mmag(ee_target: float, ee_comps: list[float]) -> float:
    """MILLIMAGNITUDES. The 1000.0 is the factor whose absence produced the Step 1b/1c error."""
    return -2.5 * math.log10(ee_target / float(np.median(ee_comps))) * 1000.0


# ---------------------------------------------------------------- expected table
def build_expected() -> dict:
    out = {"beta": BETA, "r_target_px": R_TARGET, "unit": "mmag", "table": {}}
    for r50 in R50_GRID:
        a = alpha_from_r50(r50, BETA)
        img = render_moffat((161, 161), 80.37, 80.62, a, BETA, TOTAL_FLUX, SKY_ADU)
        ee = ee_curve_photutils(img, 80.37, 80.62)
        et = ee_at(COG_RADII, ee, R_TARGET)
        row = {"fwhm_px": round(fwhm_from_r50(r50, BETA), 4),
               "ee_target": round(et, 6)}
        for k, radii in SUBSETS.items():
            ec = [ee_at(COG_RADII, ee, r) for r in radii]
            row[k] = round(delta_ap_mmag(et, ec), 2)
        out["table"][f"{r50:.2f}"] = row
    lo, hi = f"{R50_GRID[0]:.2f}", f"{R50_GRID[-1]:.2f}"
    out["range_over_span"] = {
        k: round(out["table"][hi][k] - out["table"][lo][k], 2) for k in SUBSETS
    }
    out["t4_ratio"] = round(out["range_over_span"]["G_8_9"] / out["range_over_span"]["G_gt_11"], 3)
    return out


def build_target_radius_sweep() -> dict:
    """Range over r50 span vs proxy/target radius (Step 1g configuration diagnostic)."""
    sweep_radii = [1.916, 2.416, 2.866, 3.016, 3.166]
    out: dict = {"beta": BETA, "unit": "mmag", "ranges_over_span": {}, "t4_ratio": {}}
    for r_t in sweep_radii:
        by_subset: dict[str, list[float]] = {k: [] for k in SUBSETS}
        for r50 in R50_GRID:
            a = alpha_from_r50(r50, BETA)
            img = render_moffat((161, 161), 80.37, 80.62, a, BETA, TOTAL_FLUX, SKY_ADU)
            ee = ee_curve_photutils(img, 80.37, 80.62)
            et = ee_at(COG_RADII, ee, r_t)
            for k, radii in SUBSETS.items():
                ec = [ee_at(COG_RADII, ee, r) for r in radii]
                by_subset[k].append(delta_ap_mmag(et, ec))
        ranges = {k: round(by_subset[k][-1] - by_subset[k][0], 1) for k in SUBSETS}
        out["ranges_over_span"][f"{r_t:.3f}"] = ranges
        g89, g11 = ranges["G_8_9"], ranges["G_gt_11"]
        out["t4_ratio"][f"{r_t:.3f}"] = round(g89 / g11, 2) if abs(g11) > 0.1 else None
    return out


# ---------------------------------------------------------------- gates
def gate_g0_renderer() -> tuple[bool, str]:
    a = alpha_from_r50(1.87, BETA)
    prev = None
    for o in (11, 21, 31):
        img = render_moffat((161, 161), 80.37, 80.62, a, BETA, TOTAL_FLUX, SKY_ADU, oversample=o)
        closure = float(np.sum(img - SKY_ADU)) / TOTAL_FLUX
        v = ee_at(COG_RADII, ee_curve_photutils(img, 80.37, 80.62), R_TARGET)
        if abs(closure - 1.0) > 2e-3:
            return False, f"flux closure {closure:.5f} at oversample {o}"
        if prev is not None and abs(v - prev) > 1e-3:
            return False, f"EE not converged: {prev:.5f} -> {v:.5f}"
        prev = v
    return True, f"flux closure 1.000, EE converged to {prev:.5f}"


def gate_g1_units() -> tuple[bool, str]:
    """A metric that is 1000x wrong cannot pass this. Halving EE_target is exactly 752.6 mmag."""
    d = delta_ap_mmag(0.25, [0.50])
    expect = 2.5 * math.log10(2.0) * 1000.0
    ok = abs(d - expect) < 0.01
    return ok, f"delta_ap(0.25 vs 0.50) = {d:.2f} mmag, expected {expect:.2f}"


def gate_g2_l1_vs_l2() -> tuple[bool, str]:
    """L1 and L2 must differ by 2-3 pp at r=1.916. Agreement would mean the render is wrong."""
    a = alpha_from_r50(1.87, BETA)
    img = render_moffat((161, 161), 80.37, 80.62, a, BETA, TOTAL_FLUX, SKY_ADU)
    l2 = ee_at(COG_RADII, ee_curve_photutils(img, 80.37, 80.62), R_TARGET)
    l1 = ee_moffat(R_TARGET, a, BETA) / ee_moffat(NORM_RADIUS, a, BETA)
    d = l1 - l2
    return (0.015 < d < 0.040), f"L1-L2 = {d:+.5f} at r=1.916 (pixel integration; expected 0.015-0.040)"


def gate_g3_position_invariance() -> tuple[bool, str]:
    a = alpha_from_r50(1.87, BETA)
    vals = []
    for dx, dy in [(0.0, 0.0), (0.13, 0.41), (0.37, 0.62), (0.5, 0.5), (-0.29, 0.08)]:
        img = render_moffat((161, 161), 80 + dx, 80 + dy, a, BETA, TOTAL_FLUX, SKY_ADU)
        vals.append(ee_at(COG_RADII, ee_curve_photutils(img, 80 + dx, 80 + dy), R_TARGET))
    spread = (max(vals) - min(vals)) / float(np.mean(vals))
    return spread < 0.01, f"EE spread over 5 sub-pixel positions = {100*spread:.2f} % (limit 1 %)"


def gate_g4_integer_centre_bias() -> tuple[bool, str]:
    """Measures, not asserts, the bias of the Step 1b centring method."""
    a = alpha_from_r50(1.87, BETA)
    truth, harness = [], []
    for dx, dy in [(0.0, 0.0), (0.25, 0.25), (0.5, 0.0), (0.5, 0.5), (-0.4, 0.3)]:
        img = render_moffat((161, 161), 80 + dx, 80 + dy, a, BETA, TOTAL_FLUX, SKY_ADU)
        truth.append(ee_at(COG_RADII, ee_curve_photutils(img, 80 + dx, 80 + dy), R_TARGET))
        harness.append(ee_at(COG_RADII, ee_curve_integer_centre(img, 80 + dx, 80 + dy), R_TARGET))
    bias = 100.0 * (np.mean(harness) - np.mean(truth)) / np.mean(truth)
    jitter = abs(2.5 * math.log10(max(harness) / min(harness))) * 1000.0
    return True, f"integer-centre bias {bias:+.2f} % of EE; position jitter {jitter:.0f} mmag"


def gate_g5_structure(exp: dict) -> tuple[bool, str]:
    r = exp["t4_ratio"]
    return 5.0 <= r <= 15.0, f"T4 ratio G_8_9 / G_gt_11 = {r:.2f} (band 5-15)"


PLACEMENT_SENSITIVITY_EE = {
    "note": "Synthetic Moffat at anchor r50=1.87 px: EE(1.916) vs aperture offset from true centre (Step 1i). Placement can only decrease EE.",
    "rows": [
        {"offset_px": 0.0, "EE_1.916": 0.4916, "delta_EE": 0.0, "pct_of_physics_band": 0.0},
        {"offset_px": 0.25, "EE_1.916": 0.4871, "delta_EE": -0.0045, "pct_of_physics_band": -3.0},
        {"offset_px": 0.50, "EE_1.916": 0.4737, "delta_EE": -0.0179, "pct_of_physics_band": -11.0},
        {"offset_px": 1.00, "EE_1.916": 0.4280, "delta_EE": -0.0636, "pct_of_physics_band": -39.0},
        {"offset_px": 2.00, "EE_1.916": 0.2748, "delta_EE": -0.2168, "pct_of_physics_band": -134.0},
        {"offset_px": 3.00, "EE_1.916": 0.1313, "delta_EE": -0.3603, "pct_of_physics_band": -222.0},
    ],
    "physics_band_width": 0.162,
}

SKY_BIAS_SENSITIVITY_EE = {
    "note": "Fixture L2 render at r50=1.87 px: EE(1.916) vs uniform sky bias added to annulus estimate (Step 1i). Only mechanism that increases EE above 0.65.",
    "F_12_at_zero_bias_adu": 199376.0,
    "rows": [
        {"sky_bias_adu_per_px": 0, "F_12_adu": 199376, "EE_1.916": 0.4916},
        {"sky_bias_adu_per_px": 50, "F_12_adu": 176756, "EE_1.916": 0.5512},
        {"sky_bias_adu_per_px": 100, "F_12_adu": 154137, "EE_1.916": 0.6284},
        {"sky_bias_adu_per_px": 200, "F_12_adu": 108898, "EE_1.916": 0.8788},
        {"sky_bias_adu_per_px": 400, "F_12_adu": 18420, "EE_1.916": 5.07},
    ],
}


CONTAMINATION_SENSITIVITY_MMAG = {
    "note": (
        "Synthetic pair shifts in delta_ap at r=1.916 px (Step 1h isolation audit reference). "
        "Step 1h D3: zero catalogue neighbours within 8 px and dG<5 for all five proxies "
        "and all six G 8-9 comparisons on anchor draft_435."
    ),
    "columns_flux_fraction": [0.05, 0.10, 0.30],
    "rows_separation_px": [
        {"sep_px": 1.0, "delta_ap_mmag": [3.6, 6.9, 17.9]},
        {"sep_px": 2.0, "delta_ap_mmag": [9.9, 19.1, 51.0]},
        {"sep_px": 4.0, "delta_ap_mmag": [6.7, 13.3, 38.6]},
        {"sep_px": 8.0, "delta_ap_mmag": [0.2, 0.4, 1.2]},
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true", help="print expected values as JSON")
    args = ap.parse_args()

    exp = build_expected()
    if args.emit:
        payload = {
            "expected": exp,
            "target_radius_sweep": build_target_radius_sweep(),
            "placement_sensitivity_ee": PLACEMENT_SENSITIVITY_EE,
            "sky_bias_sensitivity_ee": SKY_BIAS_SENSITIVITY_EE,
            "contamination_sensitivity_mmag": CONTAMINATION_SENSITIVITY_MMAG,
        }
        print(json.dumps(payload, indent=2))
        return 0

    gates = [
        ("G0 renderer convergence + flux closure", gate_g0_renderer()),
        ("G1 ABSOLUTE UNITS (mmag)             ", gate_g1_units()),
        ("G2 L1 vs L2 pixel-integration offset ", gate_g2_l1_vs_l2()),
        ("G3 L2 sub-pixel position invariance  ", gate_g3_position_invariance()),
        ("G4 integer-centre bias (measurement) ", gate_g4_integer_centre_bias()),
        ("G5 structural sub-ensemble ratio     ", gate_g5_structure(exp)),
    ]
    print("=" * 78)
    print("VYVAR closure A-1 reference fixture")
    print("=" * 78)
    failed = 0
    for name, (ok, msg) in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {msg}")
        failed += 0 if ok else 1

    print("\nEXPECTED delta_ap [mmag], target at r = 1.916 px, Moffat beta = 3.0")
    print(f"  {'r50':>6} {'FWHM':>7} {'EE_t':>7} " + "".join(f"{k:>10}" for k in SUBSETS))
    for r50 in R50_GRID:
        row = exp["table"][f"{r50:.2f}"]
        print(f"  {r50:6.2f} {row['fwhm_px']:7.3f} {row['ee_target']:7.4f} "
              + "".join(f"{row[k]:10.1f}" for k in SUBSETS))
    print("\n  range over r50 span: " + ", ".join(
        f"{k} {v:+.1f} mmag" for k, v in exp["range_over_span"].items()))
    print(f"\n  {failed} gate(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
