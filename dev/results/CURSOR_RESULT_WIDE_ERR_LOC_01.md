# CURSOR RESULT - WIDE-ERR-LOC-01

Date: 2026-08-14
Register ID: WIDE-ERR-LOC-01
Follows: NOISE-FLOOR-01 (accepted)
Session closing task. Report only; exported error bars not changed.

Machine: `dev/results/WIDE_ERR_LOC_01_results.json`

---

## Verdict

Item A closes: the variance expression is exact once `corr_factor` is applied
only to extended terms. Item B: a full-range fit gives `a=7.45+/-0.22`,
`b=2.20+/-0.07`, but weighted r2=0.42 -- the deficit is **not** separable into
constant source and sky factors over magnitude (W-R1). Item C: 0.10 px radius
drift is builder-versus-product near-tie on a flat SNR surface; does not block
Stage 2 beyond C2-R2. This task does **not** fix the error model.

---

## Item A -- variance accounting

**Expression (exact):**

```
sky_factor = 1 + n_pix / n_B
var_source = F / g
var_sky    = n_pix * (sky / g) * sky_factor     # sky_factor already inside
var_dark   = n_pix * (dark_e / g)
var_rn     = n_pix * (RN / g)^2
var_dig    = n_pix * (q^2 / 12)
var_ext    = (var_sky + var_dark + var_rn + var_dig) * corr_factor
var_total  = var_source + var_ext
```

`corr_factor` does **not** multiply `var_source`.

**Demonstration (draft 512 numbers from NOISE_FLOOR_01):**

| bin | reported var_total | recomputed from components | abs diff |
|-----|-------------------:|---------------------------:|---------:|
| G10.0 | 56154.043824891836 | 56154.043824891836 | 0 |
| G12.5 | 21681.158423582026 | 21681.158423582026 | 0 |

Architect reconstruction that applied `corr_factor` to all terms (including
source) yields ~63869 (G10) and ~22477 (G12.5) -- that is the 14% / 4%
discrepancy. No missing physics term; the report omitted that source is
uncorrelated. Diagnostic `terms` dict now carries `var_ext` and the expression
string.

**W-R0:** accounting closes; Item B is not labelled provisional for that reason.
(Floor remains an UL from NOISE-FLOOR-01; bright-bin R is still fragile.)

---

## Item B -- source vs sky factors

Equation fitted over all usable NP bins (n>=8), draft 512:

```
a * f_source + b * f_sky_ext = (phot_obs / phot_model)^2
f_source = var_source / var_total
f_sky_ext = (var_sky * corr_factor) / var_total
```

Weighted least squares by bin n. Floor = NOISE-FLOOR-01 UL (6.79 mmag).

| | value | uncertainty |
|-|------:|------------:|
| a (source) | 7.45 | 0.22 |
| b (sky) | 2.20 | 0.07 |
| weighted r2 | 0.42 | |
| rms resid (R2) | 0.89 | |
| n_bins / dof | 10 / 8 | |

**W-R1 FIRED.** r2=0.42: a two-parameter constant (a,b) model does **not**
describe the magnitude dependence. The deficit is not separable into fixed
source and sky factors over G8-13. That rules out the simplest explanation.

Residuals (R2_obs - R2_pred) are large at the bright end (G8.75: -4.4) where
the UL floor makes phot_obs unstable, and remain structured through mid-range.

**Sensitivity only (not the primary result):** excluding floor-dominated bins
(phot_obs/obs < 0.5; G8.25 and G8.75) gives a~10.6, b~1.56, r2~0.84 -- same
order as the architect prior (a~9, b~1.5). Primary measurement wins (W-R3);
prior was scale-only and is not a validation target.

**SNR-GATE-01 cross-check.** Implied gain 2.94 vs equipment 3.17 (~7% sky-side
agreement). Full-range b=2.20 is **not** near 1: the LC-scatter sky factor and
the empty-sky gain test **disagree**. That disagreement is a result. (Sensitivity
b~1.56 is closer but still not the SNR-GATE 7% story, and is not primary.)

**Physical implication (named, not tested).** a much larger than b, and excess
growing where source share is larger, is the signature of a term that scales
closer to flux^2 than to flux (multiplicative / relative error), not a pure
counting (Poisson) shortfall. Candidates consistent with that shape, stopped
before testing (scope freeze):

1. Flat-field residual errors (multiplicative on local flat).
2. Residual transparency / extinction variations beyond the fitted floor.
3. Time-variable enclosed-energy / aperture losses (seeing, focus, colour-PSF).
4. Scintillation / speckle beyond the Young/Osborn term already in the floor.
5. Incorrect gain on the source branch alone is **not** favoured: it would need
   g_true ~ g/a ~ 0.4 e-/ADU, absurd next to SNR-GATE sky gain.

**Register one-liner:**

> WIDE-ERR remains in the photon/sky variance channel after the completed
> Howell model; a full-range two-parameter source/sky split does not describe
> the magnitude dependence (weighted r2~0.42), so the deficit is not a single
> pair of constant source and sky factors (W-R1).

**Not measured:** per-mechanism amplitudes (explicitly out of scope).

---

## Item C -- 0.10 px aperture radius drift

**Cause:** builder-versus-product. Recomputing `compute_snr_optimal_aperture_table`
with the archived FWHM/sky/gain/RN metadata is deterministic under current code
and returns bright-end radii 0.10 px larger (two `r_step=0.05` grid steps).
SNR(r) at G7 is extremely flat near optimum (relative delta ~1e-5 between
5.161 and 5.261 px). Faint bins match at 0.0 because both hit `r_min`. Not
non-deterministic; not caused by NOISE-FLOOR-01 (`photometry_core` untouched).

**Which is correct:** recompute = current builder with recorded inputs; archived
table = what photometry used (`dao_flux` identity). Near-tie on a flat SNR
surface; treat as stale-product vs current-builder, not a physics error.

**Stage 2:** does **not** add a block. Stage 2 remains blocked under C2-R2
anyway. Stage 2 would consume photometry radii from products (archived path).

---

## What this push leaves open / invalidated

- `--full` anchor and P1 golden ledger (stale since SKY-CLIP-01 and SNR-GATE-01)
- draft 510 and 512 checksum manifests
- draft 512 products (dirty tree + broken prematch era)
- every draft built since `c9e1f8f` with shallow MASTERSTAR depth
- COMP-POOL-01 Stage 2 (blocked under C2-R2 pending guard)
- WIDE-ERR -- **diagnosed and localized, not fixed** (exported bars unchanged)
- SNR-DEPTH-01, INV-PIXELS-01, U-SKY-FALLBACK-01, A-1-OVERRIDE, D1-2 exposure
  ramp, C-EXPORT-GAP, W6-PROP, D10-1, D11-1, D5-1, Decision (4)

---

## Pre-registered rules

- **W-R0:** accounting closes; Item B not provisional for imbalance.
- **W-R1:** FIRED -- two-param model fails (r2=0.42).
- **W-R2:** FIRED -- no tuned parameter, no free scale, no error-model fix.
- **W-R3:** FIRED -- measurement over architect prior.

---

## Register diff

| ID | change |
|----|--------|
| **WIDE-ERR-LOC-01** | NEW CLOSED (report): A closes; B W-R1; C drift named |
| **WIDE-ERR** | DIAGNOSED+LOCALIZED (photon/sky; not separable to constant a,b); OPEN for fix |
| **COMP-POOL** Stage 2 | still blocked C2-R2; 0.10 px not an added block |

---

## Files

- `src_py/comp_pool_noise.py` -- Item A expression/`var_ext` in diagnostic terms
- `dev/tests/test_noise_floor_01.py` -- accounting unit test
- `dev/results/WIDE_ERR_LOC_01_results.json`
- `dev/results/CURSOR_RESULT_WIDE_ERR_LOC_01.md` (this memo)
- `docs/VYVAR_AUDIT_2026_REGISTER.md`

## Commit / --fast

- Tip: **fc6fcadf9037b0bf32a029bf6feecc5026a1c776**
- `python dev/scripts/session_baseline_check.py --fast`: **OVERALL PASS**
  (1356 passed, 27 skipped) on that tip.
- **Awaiting Milan authorization to push.** Nothing pushed.
