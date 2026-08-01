CURSOR RESULT - 2026-08-01 21:10 UTC+2

**Outcome M-x** (neither pre-registered hypothesis). Normalised slopes: flux **-0.280**, dao_flux
**-0.280**, flux_small **-0.279**, flux_large **-0.273**. Measured aperture contribution
`slope(log10 EE(r_ap) vs G)` = **-0.016**.

What I did
COG-normalised production flux using measured per-star growth curves from the Step 1f cache
(M1); retired the collinear joint fit with VIF on record (M2); verified `fwhm_estimate_px`
frame variation (M3); reconciled r50 span vs Step 1b (M4). Diagnostic only.

## M1 -- COG normalise and refit

Source: `dev/tools/closure_step1m_cog_normalise_flux.py`, `tmp/closure_step1f_ee_cache.npz`,
4058 star-frames (same admissibility as Step 1l K3).

Normalisation: `log10(F_corr) = log10(F) - log10(EE_measured(r))` with EE from each star's own
measured COG on that frame, referenced at r = 12 px.

### Measured aperture contribution (requirement 1)

| quantity | slope vs G | se | n |
|----------|------------|-----|---|
| **log10(EE_measured(r_ap))** | **-0.016** | 0.00043 | 4058 |
| log10(flux) raw | -0.296 | 0.0022 | 4058 |

The SNR table assigns larger `aperture_r_px` to brighter stars (corr = -0.95) but **EE at that
radius is nearly flat vs G** (slope -0.016). The toy -0.107 dex/mag contribution from Section 0
does not appear in the measured curves. The table tracks enclosed fraction, not raw radius alone.

### Normalised slopes (requirement 2)

| column | r used | slope raw | slope COG-normalised |
|--------|--------|-----------|----------------------|
| flux | aperture_r_px | -0.296 | **-0.280** |
| dao_flux | aperture_r_px | -0.296 | **-0.280** |
| flux_small | 3.593 px (1.5 x 2.395) | -0.271 | **-0.279** |
| flux_large | 9.58 px (4.0 x 2.395) | -0.269 | **-0.273** |

All four agree within **0.008** after normalisation. Normalisation is self-consistent.

### Pre-registered predictions

| hypothesis | predicted | measured |
|------------|-----------|----------|
| **H-ap** (aperture is whole cause) | -0.400 | **-0.280** -- **rejected** |
| **H-2** (second defect, slope ~ -0.19) | ~ -0.19 | **-0.280** -- **rejected** |

Normalisation shifts slope by only **+0.016** (from -0.296 to -0.280), matching the measured
EE(r_ap) vs G slope. Neither hypothesis is supported. Compression **persists** at ~0.28 dex/mag
below the -0.4 expectation (gap ~0.12 dex/mag).

## M2 -- collinear joint fit retired

| quantity | value |
|----------|-------|
| corr(G, aperture_r_px) | **-0.948** |
| R^2(G -> aperture_r) | 0.898 |
| **VIF(aperture_r)** | **9.80** |

Step 1l joint G slope +0.048 is instability artefact (VIF ~ 10), not physical. M1 replaces
this test. Joint OLS is not used as evidence in this step.

## M3 -- fwhm_estimate_px per-frame variation

Step 1l called this a per-star SNR-table constant. **Wrong for frame variation.**

| statistic | value |
|-----------|-------|
| stars with 139 frames | 35 / 35 |
| stars with **unique value every frame** | **35** (0 constant) |
| example G 8.18 star | 3.130 - 3.419 px across frames |

`fwhm_estimate_px` **varies by frame** and is a legitimate per-frame seeing proxy. Step 1l
residual correlation **r = -0.496** (production, G-only fit) must stand -- not dismissable as
a catalogue constant. It mixes star-level and frame-level structure; it is not yet a clean
seeing discriminator (cf. L1 r50_frame r = -0.024 on the same sample).

## M4 -- r50 range reconciliation

| source | min | median | max | method |
|--------|-----|--------|-----|--------|
| Step 1b B.1 | **1.464** | 1.873 | **1.970** | integer-centre COG, proc x,y, all 35 stars |
| Step 1f cache | **1.631** | 1.781 | **1.912** | photutils COG, Gaussian centroid, C1-admissible only |

Mean delta (cache - step1b): **-0.079 px**; max |delta| **0.232 px** per frame.

**Cause:** (1) photutils vs integer-centre COG; (2) C1 admissibility subset for r50 median;
(3) Gaussian centroid vs proc CSV positions. **Not** fixture annulus geometry (three annulus
geometries affect sky/EE measurement, not r50 from the COG curve itself). Quote r50 span with
its source; do not mix step1b and step1f numbers in one slope-x-span product.

## D5-2 -- consolidated

**Confirmed.** Pipeline flux does not scale as 10^(-0.4 G):

- Raw: **-0.296** (flux and dao_flux)
- COG-normalised: **-0.280** (all four flux columns)

**Mechanism: open.** Aperture-radius coupling via measured EE removes only 0.016 of the 0.104
dex/mag shortfall. Neither H-ap nor H-2 closes it. D5-2 is not folded into A-1. D1-2 stays
**DEFERRED** (M-2 alone would not re-open it; no unconfounded non-linearity test produced).

## Step 1l L-a mechanism

**Withdrawn.** COG normalisation does not return slope to -0.400; measured EE(r_ap) vs G is
flat (-0.016). Section 0 directional argument (brighter -> larger aperture -> steeper slope)
does not apply when the SNR table holds EE approximately constant. The joint-fit conclusion
from Step 1l is also retired (M2 VIF).

## What Step 1n must change

1. Do not consolidate `delta_ap` until the ~0.12 dex/mag residual compression has a named cause.
2. Any flux-vs-G test must use M1 COG normalisation as baseline, not raw flux.
3. Investigate frame-varying `fwhm_estimate_px` correlation (M3) separately from r50_frame.
4. Label r50 source (step1b vs step1f) in all span products.

## Docs impact

- Register: D5-2 mechanism open; L-a withdrawn; Step 1m M-x
- AUDIT_FINAL: D5-2 consolidated; D1-2 DEFERRED; D5-1 cross-ref only (not closed)
- STATE / ROADMAP: Step 1m complete

## Errors (if any)

None.

## Files changed

- `dev/tools/closure_step1m_cog_normalise_flux.py` (new)
- `dev/results/CURSOR_RESULT_closure_step1m.md` (this file)
- `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md`
- `docs/VYVAR_AUDIT_FINAL.md`
- `docs/VYVAR_STATE.md`
- `docs/VYVAR_ROADMAP.md`
